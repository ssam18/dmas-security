"""
Lightweight P2P UDP Multicast Communication Layer.

Handles DMAS message serialization and transport.  All messages are
kept under 256 bytes per the paper's constraint.

Message types:
  HEARTBEAT       — Periodic liveness announcement (1 Hz)
  THREAT_ALERT    — Broadcast when local anomaly detected
  VOTE_REQUEST    — Request peer validation of suspected threat
  VOTE_RESPONSE   — Cast vote on threat severity
  CONSENSUS_ACHIEVED — Announce decision and trigger response

Paper reference — Section III-C.
"""

from __future__ import annotations
import asyncio
import json
import logging
import socket
import struct
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    HEARTBEAT = 1
    THREAT_ALERT = 2
    VOTE_REQUEST = 3
    VOTE_RESPONSE = 4
    CONSENSUS_ACHIEVED = 5


@dataclass
class DMASMessage:
    """
    Envelope for all DMAS P2P messages.

    Serialized as compact JSON; real production would use protobuf for
    tighter size control.
    """
    msg_type: int           # MessageType value
    sender_id: str
    timestamp: float
    payload: Dict           # type-specific payload

    def to_bytes(self) -> bytes:
        raw = json.dumps({
            "t": self.msg_type,
            "s": self.sender_id,
            "ts": round(self.timestamp, 3),
            "p": self.payload,
        }, separators=(",", ":")).encode("utf-8")
        if len(raw) > 256:
            logger.warning("Message %d from %s is %d bytes (>256)",
                           self.msg_type, self.sender_id, len(raw))
        return raw

    @classmethod
    def from_bytes(cls, data: bytes) -> "DMASMessage":
        obj = json.loads(data.decode("utf-8"))
        return cls(
            msg_type=obj["t"],
            sender_id=obj["s"],
            timestamp=obj["ts"],
            payload=obj["p"],
        )

    @classmethod
    def heartbeat(cls, sender_id: str) -> "DMASMessage":
        return cls(MessageType.HEARTBEAT, sender_id, time.time(), {})

    @classmethod
    def vote_request(cls, sender_id: str, threat_id: str,
                     local_score: float, feature_vector) -> "DMASMessage":
        return cls(MessageType.VOTE_REQUEST, sender_id, time.time(), {
            "tid": threat_id,
            "score": round(local_score, 4),
            "fv": [round(f, 4) for f in feature_vector[:16]],  # cap to 16 floats
        })

    @classmethod
    def vote_response(cls, sender_id: str, threat_id: str,
                      vote_weight: float, raw_score: float) -> "DMASMessage":
        return cls(MessageType.VOTE_RESPONSE, sender_id, time.time(), {
            "tid": threat_id,
            "vw": round(vote_weight, 4),
            "rs": round(raw_score, 4),
        })

    @classmethod
    def consensus_achieved(cls, sender_id: str, threat_id: str,
                           theta_agg: float, action: str) -> "DMASMessage":
        return cls(MessageType.CONSENSUS_ACHIEVED, sender_id, time.time(), {
            "tid": threat_id,
            "agg": round(theta_agg, 4),
            "act": action,
        })


class P2PProtocol:
    """
    UDP multicast transport for DMAS agents.

    In simulation mode (use_multicast=False), messages are dispatched
    via an in-process callback bus so that a full testbed can run on a
    single machine without networking.

    Parameters
    ----------
    agent_id : str
    on_message : callable
        Callback invoked with (DMASMessage,) for each received message.
    multicast_group : str
    port : int
    use_multicast : bool
        True for real network operation; False for in-process simulation.
    """

    def __init__(
        self,
        agent_id: str,
        on_message: Callable[[DMASMessage], None],
        multicast_group: str = "224.0.0.251",
        port: int = 5355,
        use_multicast: bool = False,    # False = simulation mode
    ) -> None:
        self.agent_id = agent_id
        self.on_message = on_message
        self.multicast_group = multicast_group
        self.port = port
        self.use_multicast = use_multicast
        self._sock: Optional[socket.socket] = None
        self._running = False

        # Simulation bus: shared dict of agent_id -> P2PProtocol instance
        self._sim_bus: Optional[Dict[str, "P2PProtocol"]] = None

    # ------------------------------------------------------------------
    # Simulation mode (in-process message passing)
    # ------------------------------------------------------------------

    def attach_sim_bus(self, bus: Dict[str, "P2PProtocol"]) -> None:
        """Register on the shared simulation bus."""
        bus[self.agent_id] = self
        self._sim_bus = bus

    def sim_broadcast(self, msg: DMASMessage) -> None:
        """Deliver message to all agents on the simulation bus."""
        if self._sim_bus is None:
            logger.warning("[%s] sim_broadcast called but no bus attached", self.agent_id)
            return
        for peer_id, peer in self._sim_bus.items():
            if peer_id != self.agent_id:
                peer.on_message(msg)

    def sim_unicast(self, target_id: str, msg: DMASMessage) -> bool:
        """Deliver message to a specific agent on the simulation bus."""
        if self._sim_bus and target_id in self._sim_bus:
            self._sim_bus[target_id].on_message(msg)
            return True
        return False

    # ------------------------------------------------------------------
    # Real UDP multicast mode
    # ------------------------------------------------------------------

    async def start_async(self) -> None:
        """Start receiving UDP multicast messages (async version)."""
        if not self.use_multicast:
            self._running = True
            return
        loop = asyncio.get_event_loop()
        self._sock = self._create_recv_socket()
        self._running = True
        loop.create_task(self._recv_loop())

    async def send_async(self, msg: DMASMessage) -> None:
        """Send a message (real multicast or simulation bus)."""
        if not self.use_multicast:
            self.sim_broadcast(msg)
            return
        data = msg.to_bytes()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.sendto(data, (self.multicast_group, self.port))
        sock.close()

    def send(self, msg: DMASMessage) -> None:
        """Synchronous send (simulation mode only)."""
        if self.use_multicast:
            raise RuntimeError("Use send_async() in multicast mode")
        self.sim_broadcast(msg)

    def stop(self) -> None:
        self._running = False
        if self._sock:
            self._sock.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_recv_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        mreq = struct.pack("4sL", socket.inet_aton(self.multicast_group),
                           socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.setblocking(False)
        return sock

    async def _recv_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                data = await loop.sock_recv(self._sock, 4096)
                msg = DMASMessage.from_bytes(data)
                if msg.sender_id != self.agent_id:
                    self.on_message(msg)
            except Exception:
                await asyncio.sleep(0.001)

    def __repr__(self) -> str:
        mode = "multicast" if self.use_multicast else "simulation"
        return f"P2PProtocol(agent={self.agent_id}, mode={mode})"
