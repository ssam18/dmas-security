#!/usr/bin/env bash
# push_to_github.sh
# =================
# Initialises a local git repo and pushes to DMAS-IIoT/dmas-security.
#
# Usage:
#   chmod +x push_to_github.sh
#   GH_TOKEN=<your_personal_access_token> ./push_to_github.sh
#
# How to create a Personal Access Token (PAT):
#   1. Go to https://github.com/settings/tokens/new
#   2. Give it a name (e.g. "dmas-push")
#   3. Select scope: repo (full control of private repositories)
#   4. Click "Generate token" and copy the value
#   5. Run: GH_TOKEN=ghp_xxxx ./push_to_github.sh

set -e

REPO="https://github.com/DMAS-IIoT/dmas-security.git"
BRANCH="main"

if [ -z "$GH_TOKEN" ]; then
  echo "ERROR: GH_TOKEN environment variable is not set."
  echo "Usage: GH_TOKEN=ghp_xxxx ./push_to_github.sh"
  exit 1
fi

# Configure git if not already set
git config user.email "ssam3003@gmail.com" 2>/dev/null || true
git config user.name "Samaresh Kumar Singh" 2>/dev/null || true

cd "$(dirname "$0")"

if [ ! -d ".git" ]; then
  echo "[1/5] Initialising git repository..."
  git init
  git checkout -b "$BRANCH"
else
  echo "[1/5] Git repository already initialised."
fi

echo "[2/5] Staging all files..."
git add -A

echo "[3/5] Creating initial commit..."
git commit -m "feat: initial DMAS POC implementation

Implements the full architecture from:
  'Decentralized Multi-Agent Swarms for Autonomous Grid Security
   in Industrial IoT: A Consensus-based Approach'
  Samaresh Kumar Singh, Joyjit Roy

Components:
  - EWMADetector (λ=0.05, 3σ anomaly threshold)
  - BehavioralModel (2-layer GRU, hidden_dim=64, numpy fallback)
  - SignatureMatcher (Aho-Corasick, 14 built-in IIoT signatures)
  - MonitoringEngine (weighted ensemble θ = w_s·θ_s + w_b·θ_b + w_m·θ_m)
  - ReputationTracker (EWMA ρ update, 24h probation)
  - CVTProtocol (4-phase: detect→vote→aggregate→decide, Eqs 2-3)
  - P2PProtocol (UDP multicast + in-process simulation bus)
  - ResponseExecutor (quarantine, JSONL alerting, forensics)
  - DMASAgent (top-level orchestrator)
  - DMASTestbed (in-process multi-agent simulation)
  - AttackGenerator (6 attack types: DDoS, MitM, Replay, Injection, Malware, Zero-day)

Tests: 53 passing
Demos: single_agent, multi_agent, byzantine_stress_test
" 2>/dev/null || echo "[3/5] Nothing new to commit."

echo "[4/5] Setting remote origin..."
REMOTE_URL="https://${GH_TOKEN}@github.com/DMAS-IIoT/dmas-security.git"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

echo "[5/5] Pushing to $BRANCH..."
git push -u origin "$BRANCH" --force

echo ""
echo "✅  Successfully pushed to https://github.com/DMAS-IIoT/dmas-security"
echo "    Branch: $BRANCH"
