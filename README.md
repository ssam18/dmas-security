# DMAS — Decentralized Multi-Agent Swarms for Autonomous Grid Security

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Proof-of-Concept implementation of the DMAS architecture described in:

> **Decentralized Multi-Agent Swarms for Autonomous Grid Security in Industrial IoT: A Consensus-based Approach**  
> Samaresh Kumar Singh, Joyjit Roy — IEEE Senior Members

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              OPTIONAL CLOUD (Threat Intelligence)            │
└─────────────────────────────────────────────────────────────┘
           ↕ periodic model updates
┌──────────────────────────────────────────────────────────────┐
│                  DECENTRALIZED AGENT LAYER                   │
│  ┌──────────────┐  P2P UDP   ┌──────────────┐               │
│  │  Edge GW 1   │◄──────────►│  Edge GW 2   │  ...          │
│  │  DMAS Agent  │            │  DMAS Agent  │               │
│  │ ┌──────────┐ │            │ ┌──────────┐ │               │
│  │ │Monitoring│ │            │ │Monitoring│ │               │
│  │ │ EWMA+GRU │ │            │ │ EWMA+GRU │ │               │
│  │ │ +SigMatch│ │            │ │ +SigMatch│ │               │
│  │ ├──────────┤ │            │ ├──────────┤ │               │
│  │ │  CVT     │ │            │ │  CVT     │ │               │
│  │ │Consensus │ │            │ │Consensus │ │               │
│  │ ├──────────┤ │            │ ├──────────┤ │               │
│  │ │ Response │ │            │ │ Response │ │               │
│  │ │ Executor │ │            │ │ Executor │ │               │
│  │ └──────────┘ │            │ └──────────┘ │               │
│  └──────────────┘            └──────────────┘               │
└──────────────────────────────────────────────────────────────┘
           ↕ monitors
┌──────────────────────────────────────────────────────────────┐
│                   IIoT DEVICE LAYER                          │
│    PLCs    SCADA    Cameras    Sensors    Robots             │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the multi-agent simulation demo (5 agents, mixed attack traffic)
python examples/multi_agent_demo.py

# Run a single-agent anomaly detection demo
python examples/single_agent_demo.py

# Run the Byzantine fault-tolerance stress test
python examples/byzantine_stress_test.py

# Run all tests
pytest tests/ -v
```

## Project Structure

```
dmas-security/
├── dmas/
│   ├── agent.py                  # DMASSAgent — top-level orchestrator
│   ├── monitoring/
│   │   ├── ewma_detector.py      # Statistical EWMA anomaly detector
│   │   ├── behavioral_model.py   # GRU-based temporal behavioral model
│   │   ├── signature_matcher.py  # Aho-Corasick signature matching
│   │   └── monitoring_engine.py  # Ensemble score compositor
│   ├── consensus/
│   │   ├── cvt_protocol.py       # 4-phase CVT consensus algorithm
│   │   └── reputation.py         # EWMA-based agent reputation tracking
│   ├── communication/
│   │   └── p2p_protocol.py       # UDP multicast P2P messaging layer
│   ├── response/
│   │   └── response_executor.py  # Quarantine / alert / forensics
│   └── simulation/
│       ├── testbed.py            # In-process multi-agent testbed
│       └── attack_generator.py   # Synthetic attack traffic generator
├── tests/
│   ├── test_ewma.py
│   ├── test_cvt.py
│   ├── test_reputation.py
│   ├── test_monitoring_engine.py
│   └── test_simulation.py
├── examples/
│   ├── single_agent_demo.py
│   ├── multi_agent_demo.py
│   └── byzantine_stress_test.py
└── config/
    └── default_config.yaml
```

## Key Parameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tau_alert` | 0.45 | Threat score threshold to initiate CVT |
| `tau_consensus` | 0.75 | Aggregate vote threshold for quarantine |
| `delta_t_ms` | 0.5 | Vote collection timeout (ms) |
| `alpha` | 0.1 | Distance-decay rate for proximity weighting |
| `beta` | 0.9 | Reputation EWMA smoothing factor |
| `lambda_ewma` | 0.05 | EWMA smoothing for statistical detector |
| `sigma_threshold` | 3.0 | σ-threshold for EWMA anomaly flag |

## License

MIT --> see [LICENSE](LICENSE).
