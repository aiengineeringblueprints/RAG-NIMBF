"""Pass 2 (issue #46): re-score run100 traces with direct-RAGAS-like concurrency."""
import os
import yaml
import mlflow
from benchmark.mlflow_scorers import JudgeEndpoint, score_run_traces

os.environ.update({
    "MLFLOW_GENAI_EVAL_MAX_WORKERS": "8",         # rows in parallel (= eval_critic_max_workers)
    "MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS": "1",  # scorers per row sequential
    "MLFLOW_GENAI_EVAL_LLM_TIMEOUT": "300",
    "MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT": "1800",
})
mlflow.set_tracking_uri("sqlite:///mlflow.db")
names = yaml.safe_load(open("experiments/ragas-mlflow-calibration.yaml"))["settings"]["mlflow_scorers"]
endpoint = JudgeEndpoint("openai:/gemma3:12b", "http://localhost:11434/v1", "not-needed",
                         {"temperature": 0.0, "max_tokens": 4096})
print(score_run_traces("ac134264f76b4961bb1b6f877de6ce52", names, endpoint))
