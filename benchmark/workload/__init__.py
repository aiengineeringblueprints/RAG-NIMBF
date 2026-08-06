"""RAGPerf-style concurrent workload generator.

Issues a configurable mix of Query / Insert / Update / Remove ops against
a live RAG pipeline (vector store + generator) under open-loop arrival and
bounded concurrency. See ``NOTES_D_Workload.md`` at the repo root for the
full design.
"""

from benchmark.workload.distributions import uniform_sample, zipfian_sample
from benchmark.workload.operations import (
    InsertOp,
    Operation,
    QueryOp,
    RemoveOp,
    UpdateOp,
)
from benchmark.workload.generator import WorkloadConfig, WorkloadGenerator
from benchmark.workload.runner import WorkloadRunner, WorkloadSummary
from benchmark.workload.report import summarize, write_summary_json
from benchmark.workload.integration import run_workload_phase

__all__ = [
    "uniform_sample",
    "zipfian_sample",
    "Operation",
    "QueryOp",
    "InsertOp",
    "UpdateOp",
    "RemoveOp",
    "WorkloadConfig",
    "WorkloadGenerator",
    "WorkloadRunner",
    "WorkloadSummary",
    "summarize",
    "write_summary_json",
    "run_workload_phase",
]
