"""
DMAS — Decentralized Multi-Agent Swarms for Autonomous Grid Security.

POC implementation of the architecture described in:
  'Decentralized Multi-Agent Swarms for Autonomous Grid Security
   in Industrial IoT: A Consensus-based Approach'
  Samaresh Kumar Singh, Joyjit Roy
"""

from dmas.agent import DMASAgent
from dmas.monitoring.monitoring_engine import MonitoringEngine
from dmas.consensus.cvt_protocol import CVTProtocol
from dmas.consensus.reputation import ReputationTracker

__all__ = ["DMASAgent", "MonitoringEngine", "CVTProtocol", "ReputationTracker"]
__version__ = "0.1.0"
