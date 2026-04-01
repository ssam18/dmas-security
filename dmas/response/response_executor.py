"""
Response Executor.

Executes mitigation actions once CVT consensus is reached:
  1. QUARANTINE — modify firewall rules to isolate the device
  2. ALERT      — notify human operators (log to JSONL)
  3. FORENSICS  — collect and record evidence for post-incident analysis

In simulation mode, all actions are logged rather than applied to real
network infrastructure.

Paper reference — Section III-B-4.
"""

from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class QuarantineRecord:
    device_id: str
    agent_id: str
    threat_id: str
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    is_active: bool = True


@dataclass
class ForensicEntry:
    threat_id: str
    device_id: str
    agent_id: str
    theta_agg: float
    n_votes: int
    action: str
    timestamp: float = field(default_factory=time.time)
    evidence: Dict = field(default_factory=dict)


class ResponseExecutor:
    """
    Executes and logs mitigation responses.

    Parameters
    ----------
    agent_id : str
    alert_log_path : str, optional
        Path to JSONL file for alert logs.
    forensics_log_path : str, optional
        Path to JSONL file for forensics records.
    simulation_mode : bool
        If True, no real firewall rules are modified.
    """

    def __init__(
        self,
        agent_id: str,
        alert_log_path: str = "logs/alerts.jsonl",
        forensics_log_path: str = "logs/forensics.jsonl",
        simulation_mode: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.alert_log_path = alert_log_path
        self.forensics_log_path = forensics_log_path
        self.simulation_mode = simulation_mode

        self._quarantined: Dict[str, QuarantineRecord] = {}   # device_id -> record
        self._forensics: List[ForensicEntry] = []
        self._alert_count: int = 0

        os.makedirs(os.path.dirname(alert_log_path) if os.path.dirname(alert_log_path) else ".", exist_ok=True)
        os.makedirs(os.path.dirname(forensics_log_path) if os.path.dirname(forensics_log_path) else ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        action: str,
        device_id: str,
        threat_id: str,
        theta_agg: float = 0.0,
        n_votes: int = 0,
        evidence: Optional[Dict] = None,
    ) -> None:
        """
        Execute a mitigation action.

        Parameters
        ----------
        action : "QUARANTINE" | "MONITOR" | "CLEAR"
        device_id : str
        threat_id : str
        theta_agg : float
        n_votes : int
        evidence : dict, optional
        """
        ts = time.time()
        evidence = evidence or {}

        if action == "QUARANTINE":
            self._quarantine(device_id, threat_id, theta_agg, evidence)
        elif action == "CLEAR":
            self._clear(device_id, threat_id)

        # Always log an alert and forensics entry
        self._alert(action, device_id, threat_id, theta_agg, ts)
        self._record_forensics(
            ForensicEntry(
                threat_id=threat_id,
                device_id=device_id,
                agent_id=self.agent_id,
                theta_agg=theta_agg,
                n_votes=n_votes,
                action=action,
                timestamp=ts,
                evidence=evidence,
            )
        )

    # ------------------------------------------------------------------
    # Quarantine management
    # ------------------------------------------------------------------

    def _quarantine(self, device_id: str, threat_id: str,
                    theta_agg: float, evidence: Dict) -> None:
        if device_id in self._quarantined:
            logger.debug("[%s] Device %s already quarantined", self.agent_id, device_id)
            return

        rec = QuarantineRecord(
            device_id=device_id,
            agent_id=self.agent_id,
            threat_id=threat_id,
            reason=f"CVT consensus Θ_agg={theta_agg:.3f}",
        )
        self._quarantined[device_id] = rec

        if self.simulation_mode:
            logger.warning("[%s] [SIM] QUARANTINE device=%s threat=%s Θ_agg=%.3f",
                           self.agent_id, device_id, threat_id, theta_agg)
        else:
            # Production: modify firewall rules via iptables / nftables
            self._apply_firewall_block(device_id)

    def _clear(self, device_id: str, threat_id: str) -> None:
        if device_id in self._quarantined:
            self._quarantined[device_id].is_active = False
            logger.info("[%s] CLEARED quarantine for device %s (threat %s)",
                        self.agent_id, device_id, threat_id)
            if not self.simulation_mode:
                self._remove_firewall_block(device_id)

    def _apply_firewall_block(self, device_id: str) -> None:
        """Production stub: add iptables DROP rule for device IP."""
        logger.info("[%s] [REAL] iptables -A INPUT -s %s -j DROP",
                    self.agent_id, device_id)

    def _remove_firewall_block(self, device_id: str) -> None:
        logger.info("[%s] [REAL] iptables -D INPUT -s %s -j DROP",
                    self.agent_id, device_id)

    # ------------------------------------------------------------------
    # Alerting and forensics
    # ------------------------------------------------------------------

    def _alert(self, action: str, device_id: str, threat_id: str,
               theta_agg: float, ts: float) -> None:
        self._alert_count += 1
        record = {
            "seq": self._alert_count,
            "agent": self.agent_id,
            "action": action,
            "device": device_id,
            "threat_id": threat_id,
            "theta_agg": round(theta_agg, 4),
            "ts": round(ts, 3),
        }
        try:
            with open(self.alert_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.error("Alert log write failed: %s", e)
        logger.info("[%s] ALERT #%d action=%s device=%s θ_agg=%.3f",
                    self.agent_id, self._alert_count, action, device_id, theta_agg)

    def _record_forensics(self, entry: ForensicEntry) -> None:
        self._forensics.append(entry)
        try:
            with open(self.forensics_log_path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except OSError as e:
            logger.error("Forensics log write failed: %s", e)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def quarantined_devices(self) -> Set[str]:
        return {did for did, r in self._quarantined.items() if r.is_active}

    def is_quarantined(self, device_id: str) -> bool:
        rec = self._quarantined.get(device_id)
        return rec is not None and rec.is_active

    def stats(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "total_alerts": self._alert_count,
            "active_quarantines": len(self.quarantined_devices),
            "total_forensics": len(self._forensics),
        }

    def __repr__(self) -> str:
        return (f"ResponseExecutor(agent={self.agent_id}, "
                f"mode={'sim' if self.simulation_mode else 'real'}, "
                f"quarantined={len(self.quarantined_devices)})")
