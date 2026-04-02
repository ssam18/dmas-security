"""
Signature-based Anomaly Detector.

Uses an Aho-Corasick automaton for O(n) multi-pattern matching over
raw packet bytes against a database of known attack signatures.

Paper reference — Section III-B-1:
  'Contains a database of 1,247 known attack patterns sourced from
   Snort community rules (version 3.1) and custom IIoT-specific
   signatures.  Pattern matching operates on raw packet headers and
   payload bytes using a multi-pattern Aho-Corasick automaton for O(n)
   throughput.'
"""

from __future__ import annotations
from typing import Dict, List, Optional

try:
    import ahocorasick
    _AHO_AVAILABLE = True
except ImportError:
    _AHO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Built-in lightweight signatures (representative IIoT patterns)
# These cover the six attack categories evaluated in the paper.
# ---------------------------------------------------------------------------

_DEFAULT_SIGNATURES: List[Dict] = [
    # --- DDoS indicators ---
    {"id": "DDOS-001", "name": "SYN Flood header pattern",
     "pattern": b"\x02\x04\x05\xb4\x01\x01\x08\x0a", "severity": 0.8},
    {"id": "DDOS-002", "name": "ICMP flood payload",
     "pattern": b"\x08\x00\x00\x00\x00\x00\x00\x00", "severity": 0.7},

    # --- MitM indicators ---
    {"id": "MITM-001", "name": "ARP spoofing opcode",
     "pattern": b"\x00\x02\x00\x00\x00\x00\x00\x00", "severity": 0.85},
    {"id": "MITM-002", "name": "SSL stripping redirect",
     "pattern": b"HTTP/1.1 301\r\nLocation: http://", "severity": 0.9},

    # --- Replay attack indicators ---
    {"id": "REPLAY-001", "name": "Modbus function code replay",
     "pattern": b"\x00\x01\x00\x00\x00\x06\xff\x03", "severity": 0.75},
    {"id": "REPLAY-002", "name": "DNP3 replay marker",
     "pattern": b"\x05\x64", "severity": 0.7},

    # --- Injection attack indicators ---
    {"id": "INJECT-001", "name": "SQL injection pattern",
     "pattern": b"' OR '1'='1", "severity": 0.95},
    {"id": "INJECT-002", "name": "Command injection",
     "pattern": b"; /bin/sh", "severity": 0.95},
    {"id": "INJECT-003", "name": "Modbus coil write injection",
     "pattern": b"\x00\x0f\x00\x00", "severity": 0.8},

    # --- Malware propagation indicators ---
    {"id": "MALWARE-001", "name": "EternalBlue SMB exploit",
     "pattern": b"\xff\x53\x4d\x42\x72\x00\x00\x00", "severity": 0.98},
    {"id": "MALWARE-002", "name": "Mirai default credential attempt",
     "pattern": b"root\x00xc3511\x00", "severity": 0.9},
    {"id": "MALWARE-003", "name": "Reverse shell TCP marker",
     "pattern": b"bash -i >& /dev/tcp/", "severity": 0.99},

    # --- SCADA/ICS specific ---
    {"id": "ICS-001", "name": "Siemens S7 STOP command",
     "pattern": b"\x32\x01\x00\x00\x00\x00\x00\x08\x00\x00\x29", "severity": 0.99},
    {"id": "ICS-002", "name": "Unauthorized PLC write",
     "pattern": b"\x32\x07\x00\x00", "severity": 0.85},
]


class SignatureMatcher:
    """
    Multi-pattern signature matcher using Aho-Corasick (falls back to
    linear regex scan if pyahocorasick is not installed).

    Parameters
    ----------
    signatures : list of dicts, optional
        Each dict: {"id": str, "name": str, "pattern": bytes, "severity": float}.
        Defaults to the built-in IIoT signature set.
    """

    def __init__(self, signatures: Optional[List[Dict]] = None) -> None:
        self._signatures = signatures if signatures is not None else list(_DEFAULT_SIGNATURES)
        self._build_automaton()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_signature(self, sig: Dict) -> None:
        """Add a new signature at runtime."""
        self._signatures.append(sig)
        self._build_automaton()

    def score(self, payload: bytes) -> float:
        """
        Return a threat score in [0, 1].

        The score is the maximum severity among all matching signatures,
        or 0.0 if nothing matches.
        """
        if not payload:
            return 0.0
        return self._match(payload)

    def match_details(self, payload: bytes) -> List[Dict]:
        """Return all matching signatures (for forensic logging)."""
        if not payload:
            return []
        return self._match_all(payload)

    @property
    def n_signatures(self) -> int:
        return len(self._signatures)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_automaton(self) -> None:
        self._idx: Dict[bytes, int] = {
            sig["pattern"]: i for i, sig in enumerate(self._signatures)
        }
        if _AHO_AVAILABLE and self._signatures:
            self._automaton = ahocorasick.Automaton()
            for i, sig in enumerate(self._signatures):
                self._automaton.add_word(sig["pattern"].decode("latin-1"), (i, sig))
            self._automaton.make_automaton()
        else:
            self._automaton = None

    def _match(self, payload: bytes) -> float:
        max_severity = 0.0
        if self._automaton is not None:
            text = payload.decode("latin-1", errors="replace")
            for _, (_, sig) in self._automaton.iter(text):
                max_severity = max(max_severity, sig["severity"])
        else:
            for sig in self._signatures:
                if sig["pattern"] in payload:
                    max_severity = max(max_severity, sig["severity"])
        return max_severity

    def _match_all(self, payload: bytes) -> List[Dict]:
        matched = []
        if self._automaton is not None:
            text = payload.decode("latin-1", errors="replace")
            for _, (_, sig) in self._automaton.iter(text):
                matched.append(sig)
        else:
            for sig in self._signatures:
                if sig["pattern"] in payload:
                    matched.append(sig)
        return matched

    def __repr__(self) -> str:
        backend = "ahocorasick" if _AHO_AVAILABLE else "linear-scan"
        return f"SignatureMatcher(n_signatures={self.n_signatures}, backend={backend})"
