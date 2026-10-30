#!/usr/bin/env bash
# Run the Enterprise RAG Blueprint plain-RAG benchmark with its environment.
#
# Usage: scripts/run_blueprint_rag.sh <run-dir> [manifest]
#   e.g. scripts/run_blueprint_rag.sh results/Enterprise-RAG/run4
#
# Every variable below can be overridden from the calling shell. The log is
# written to ~/bp-rag-<run-dir name>.log (outside the repo).
set -euo pipefail

RUN_DIR=${1:?usage: scripts/run_blueprint_rag.sh <run-dir> [manifest]}
MANIFEST=${2:-experiments/enterprise-rag-blueprint-rag.yaml}

export RAG_ADAPTER_MODULES=${RAG_ADAPTER_MODULES:-examples.enterprise_rag_plugin.mcp_adapter}
export ENTERPRISE_RAG_BLUEPRINT_PATH=${ENTERPRISE_RAG_BLUEPRINT_PATH:-$HOME/source/Enterprise_RAG_Blueprint}
export VECTORDB_DIR=${VECTORDB_DIR:-$HOME/.cache/blueprint-bench/vectordb-rag}
export INDEX_NAME=${INDEX_NAME:-bench-blueprint-rag}
export MCP_ENABLED=${MCP_ENABLED:-false}
# HuggingFace Xet downloads can hang on a reset connection; plain HTTP works.
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

cd "$(dirname "$0")/.."
LOG=${LOG:-$HOME/bp-rag-$(basename "$RUN_DIR").log}
echo "manifest=$MANIFEST run_dir=$RUN_DIR log=$LOG"
.venv/bin/python -m benchmark.worker run "$MANIFEST" \
  --keep-going --run-dir "$RUN_DIR" 2>&1 | tee "$LOG"
