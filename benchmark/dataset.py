import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset
from rich.console import Console

from benchmark.dataset_adapters import resolve_adapter

console = Console()

RAGPERF_WIKIPEDIA_DATASET = "ragperf-wikipedia-nq"
RAGPERF_WIKIPEDIA_HF_ID = "wikimedia/wikipedia"
RAGPERF_WIKIPEDIA_CONFIG = "20231101.en"
RAGPERF_NQ_HF_ID = "sentence-transformers/natural-questions"

SAMPLE_REQUIRED_KEYS = ("question", "ground_truth", "context", "metadata")


@dataclass(frozen=True)
class BenchmarkSample:
    """Typed representation of the public benchmark sample dict shape."""

    question: str
    ground_truth: str
    context: str | list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "context": self.context,
            "metadata": self.metadata,
        }


def normalize_sample(
    sample: Mapping[str, Any], source: str = "sample"
) -> dict[str, Any]:
    """Validate and normalize a benchmark sample while preserving dict flow.

    The returned dict keeps the public ``question``, ``ground_truth``,
    ``context``, and ``metadata`` shape expected by the rest of the pipeline.
    """
    missing = [key for key in SAMPLE_REQUIRED_KEYS if key not in sample]
    if missing:
        raise ValueError(
            f"{source} is missing required benchmark sample field(s): "
            f"{', '.join(missing)}"
        )

    metadata = sample["metadata"]
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(
            f"{source}.metadata must be a dict, got {type(metadata).__name__}"
        )

    context = sample["context"]
    if isinstance(context, list):
        normalized_context: str | list[str] = [str(item) for item in context]
    else:
        normalized_context = str(context)

    normalized = BenchmarkSample(
        question=str(sample["question"]),
        ground_truth=str(sample["ground_truth"]),
        context=normalized_context,
        metadata=dict(metadata),
    ).to_dict()

    for key, value in sample.items():
        if key not in normalized:
            normalized[key] = value
    return normalized


def normalize_samples(
    samples: list[Mapping[str, Any]],
    source: str = "sample",
) -> list[dict[str, Any]]:
    return [
        normalize_sample(sample, source=f"{source}[{index}]")
        for index, sample in enumerate(samples)
    ]


def load_benchmark_data(
    dataset_name: str = "t2-ragbench",
    subset: str | None = None,
    sample_size: int = 50,
    dataset_path: str | None = None,
    question_field: str | None = None,
    ground_truth_field: str | None = None,
    context_field: str | None = None,
    metadata_field: str | None = None,
    split: str | None = None,
    max_examples: int | None = None,
) -> list[dict]:
    adapter = resolve_adapter(dataset_name)

    if dataset_name in ("jsonl", "jsonl-shared", "csv"):
        return _load_local_dataset(
            dataset_name="jsonl" if dataset_name == "jsonl-shared" else dataset_name,
            dataset_path=dataset_path,
            sample_size=sample_size,
            question_field=question_field
            or os.getenv("DATASET_QUESTION_FIELD", "question"),
            ground_truth_field=ground_truth_field
            or os.getenv("DATASET_GROUND_TRUTH_FIELD", "ground_truth"),
            context_field=context_field
            or os.getenv("DATASET_CONTEXT_FIELD", "context"),
            metadata_field=metadata_field
            or os.getenv("DATASET_METADATA_FIELD", "metadata"),
        )

    label = subset or "default"
    console.print(f"[bold blue]Loading {adapter.hf_id} ({label})...[/bold blue]")

    # For ragbench_<component> adapters the component name is the HF config
    # subset. It always takes precedence over the env-derived ``subset`` value
    # (which may leak from DATASET_SUBSET for unrelated adapters), because the
    # adapter name is the authoritative signal here.
    component = _ragbench_component_for_adapter(dataset_name)
    if component is not None:
        effective_subset = component
    else:
        effective_subset = subset

    kwargs: dict = {}
    if adapter.requires_subset and effective_subset:
        kwargs["name"] = effective_subset
    try:
        ds = load_dataset(adapter.hf_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 — surface a clear, actionable error
        raise RuntimeError(
            f"Failed to download dataset {adapter.hf_id!r}"
            + (f" (subset={effective_subset!r})" if effective_subset else "")
            + f". Network access to HuggingFace is required. Original error: {exc}"
        ) from exc

    chosen_split = (
        split
        or os.getenv("DATASET_SPLIT")
        or adapter.preferred_split
    )
    if chosen_split not in ds:
        chosen_split = list(ds.keys())[0]
    data = ds[chosen_split]

    cap = max_examples or _env_int("DATASET_MAX_EXAMPLES") or sample_size
    if cap and cap < len(data):
        data = data.shuffle(seed=42).select(range(cap))

    rows = list(data)
    _maybe_write_ragbench_trace_sidecar(dataset_name, subset, chosen_split, rows)

    samples = []
    for row in rows:
        gt_raw = row.get(adapter.ground_truth_key, "")
        if adapter.ground_truth_transform:
            gt = adapter.ground_truth_transform(gt_raw)
        else:
            gt = str(gt_raw)

        metadata = {k: row.get(k) for k in adapter.metadata_keys if k in row}
        trace = _ragbench_trace_for_row(dataset_name, row)
        if trace:
            metadata["ragbench_trace"] = trace

        samples.append(
            normalize_sample(
                {
                    "question": row[adapter.question_key],
                    "ground_truth": gt,
                    "context": adapter.build_context(row),
                    "metadata": metadata,
                },
                source=f"{dataset_name}:{chosen_split}[{len(samples)}]",
            )
        )

    console.print(
        f"[green]Loaded {len(samples)} samples from {adapter.hf_id} "
        f"({chosen_split} split)[/green]"
    )
    return samples


def load_corpus_and_questions(
    dataset_name: str = "squad",
    subset: str | None = None,
    sample_size: int = 50,
    dataset_path: str | None = None,
    corpus_path: str | None = None,
    question_field: str | None = None,
    ground_truth_field: str | None = None,
    context_field: str | None = None,
    metadata_field: str | None = None,
    split: str | None = None,
    max_examples: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load data and split into a deduplicated corpus and per-question entries.

    Returns (corpus, questions) where:
      - corpus: list of {context, metadata} dicts — all unique contexts
      - questions: list of {question, ground_truth, context, metadata} dicts
    """
    if dataset_name == RAGPERF_WIKIPEDIA_DATASET:
        return _load_ragperf_wikipedia_nq(sample_size=sample_size)

    if dataset_name == "jsonl-shared":
        samples = normalize_samples(
            load_benchmark_data(
                dataset_name,
                subset,
                sample_size,
                dataset_path=dataset_path,
                question_field=question_field,
                ground_truth_field=ground_truth_field,
                context_field=context_field,
                metadata_field=metadata_field,
                split=split,
                max_examples=max_examples,
            ),
            source=dataset_name,
        )
        return _load_local_corpus(corpus_path), samples

    samples = normalize_samples(
        load_benchmark_data(
            dataset_name,
            subset,
            sample_size,
            dataset_path=dataset_path,
            question_field=question_field,
            ground_truth_field=ground_truth_field,
            context_field=context_field,
            metadata_field=metadata_field,
            split=split,
            max_examples=max_examples,
        ),
        source=dataset_name,
    )

    seen: dict[str, str] = {}  # context text → stable gold document ID
    corpus: list[dict] = []

    for sample in samples:
        ctx = sample["context"]
        ctx_key = _context_text(ctx)
        if ctx_key not in seen:
            doc_id = _stable_doc_id(dataset_name, ctx_key, len(corpus))
            seen[ctx_key] = doc_id
            corpus.append(
                {
                    "context": ctx,
                    "metadata": _chroma_safe_metadata(
                        {
                            **sample.get("metadata", {}),
                            "doc_id": doc_id,
                        }
                    ),
                }
            )
        sample["metadata"] = {
            **sample.get("metadata", {}),
            "gold_doc_id": seen[ctx_key],
        }

    console.print(
        f"[green]Deduplicated corpus: {len(corpus)} unique documents "
        f"for {len(samples)} questions[/green]"
    )
    return corpus, samples


def _load_local_corpus(corpus_path: str | None) -> list[dict[str, Any]]:
    """Load a deterministic text/Markdown corpus for managed external RAGs."""
    value = corpus_path or os.getenv("DATASET_CORPUS_PATH")
    if not value:
        raise ValueError(
            "DATASET_CORPUS_PATH is required when DATASET_NAME=jsonl-shared"
        )
    root = Path(value).resolve()
    if not root.is_dir():
        raise ValueError(f"DATASET_CORPUS_PATH is not a directory: {root}")

    allowed = {".md", ".txt"}
    paths: list[Path] = []
    for candidate in root.rglob("*"):
        # This corpus may be uploaded to an external service. Following a
        # repository-controlled link could disclose an arbitrary local file.
        if candidate.is_symlink():
            raise ValueError(
                f"Symbolic links are not allowed in DATASET_CORPUS_PATH: {candidate}"
            )
        if not candidate.is_file() or candidate.suffix.lower() not in allowed:
            continue
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Corpus document resolves outside DATASET_CORPUS_PATH: {candidate}"
            ) from exc
        paths.append(resolved)
    paths.sort()
    if not paths:
        raise ValueError(f"No Markdown or text documents found in {root}")

    corpus: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root)
        source_id = path.stem
        corpus.append(
            {
                "context": path.read_text(encoding="utf-8"),
                "metadata": {
                    "doc_id": source_id,
                    "source_id": source_id,
                    "source_name": path.name,
                    "source_path": str(relative),
                },
            }
        )
    console.print(
        f"[green]Loaded {len(corpus)} local corpus documents from {root}[/green]"
    )
    return corpus


def _load_local_dataset(
    dataset_name: str,
    dataset_path: str | None,
    sample_size: int,
    question_field: str,
    ground_truth_field: str,
    context_field: str,
    metadata_field: str,
) -> list[dict]:
    path_value = dataset_path or os.getenv("DATASET_PATH")
    if not path_value:
        raise ValueError("DATASET_PATH is required when DATASET_NAME=jsonl or csv")

    path = Path(path_value)
    if not path.exists():
        raise ValueError(f"DATASET_PATH does not exist: {path}")

    if dataset_name == "jsonl":
        rows = _read_jsonl(path)
    elif dataset_name == "csv":
        rows = _read_csv(path)
    else:
        raise ValueError(f"Unsupported local dataset type: {dataset_name}")

    if sample_size and sample_size < len(rows):
        rows = rows[:sample_size]

    samples = []
    for index, row in enumerate(rows):
        metadata = row.get(metadata_field, {})
        if isinstance(metadata, str) and metadata.strip():
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {metadata_field: metadata}
        elif metadata in (None, ""):
            metadata = {}

        samples.append(
            normalize_sample(
                {
                    "question": _require_field(row, question_field, index),
                    "ground_truth": _require_field(row, ground_truth_field, index),
                    "context": row.get(context_field, ""),
                    "metadata": metadata,
                },
                source=f"{dataset_name}:{path}[{index}]",
            )
        )

    console.print(f"[green]Loaded {len(samples)} local samples from {path}[/green]")
    return samples


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {path}:{line_number} must be an object")
            rows.append(row)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _require_field(row: Mapping[str, Any], field: str, index: int) -> Any:
    if field not in row:
        raise ValueError(
            f"Local dataset row {index} is missing required field {field!r}"
        )
    return row[field]


def _context_text(context: str | list[str]) -> str:
    if isinstance(context, list):
        return "\n".join(context)
    return context


_CHROMA_PRIMITIVES = (str, int, float, bool)


def _chroma_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce metadata values into Chroma's accepted types.

    Chroma only stores str/int/float/bool/None and lists of those. Multi-hop
    datasets stash rich structures (raw ``context`` dicts, ``supporting_facts``
    nested lists) in sample metadata for downstream gold-scoring. Serialise
    anything richer to a JSON string so Chroma's upsert accepts it. Downstream
    readers recover the original via ``json.loads``.
    """
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, _CHROMA_PRIMITIVES):
            safe[key] = value
            continue
        if isinstance(value, list) and all(
            item is None or isinstance(item, _CHROMA_PRIMITIVES) for item in value
        ):
            safe[key] = value
            continue
        safe[key] = json.dumps(value, ensure_ascii=False)
    return safe


def _stable_doc_id(dataset_name: str, context: str, index: int) -> str:
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_name}_doc_{index}_{digest}"


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _ragbench_component_for_adapter(dataset_name: str) -> str | None:
    """Return the RAGBench component name for ``ragbench_<component>`` adapters."""
    prefix = "ragbench_"
    if not dataset_name.startswith(prefix):
        return None
    component = dataset_name[len(prefix) :]
    from benchmark.ragbench_adapter import RAGBENCH_COMPONENTS

    return component if component in RAGBENCH_COMPONENTS else None


def _ragbench_trace_for_row(dataset_name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract TRACe gold annotations for a row when ``dataset_name`` is RAGBench."""
    component = _ragbench_component_for_adapter(dataset_name)
    if component is None:
        return {}
    from benchmark.ragbench_adapter import _ragbench_trace_metadata

    return _ragbench_trace_metadata(row)


def _maybe_write_ragbench_trace_sidecar(
    dataset_name: str,
    subset: str | None,
    split: str,
    rows: list[Mapping[str, Any]],
) -> None:
    """Persist RAGBench TRACe gold annotations to a sidecar JSON when applicable."""
    component = _ragbench_component_for_adapter(dataset_name)
    if component is None:
        return
    from benchmark.ragbench_adapter import write_trace_sidecar

    cache_root = os.getenv("DATASET_CACHE_ROOT") or "datasets"
    write_trace_sidecar(cache_root, component, split, [dict(r) for r in rows])


def _load_ragperf_wikipedia_nq(sample_size: int) -> tuple[list[dict], list[dict]]:
    """Load RAGPerf-style Wikipedia corpus with Natural Questions QA pairs.

    RAGPerf indexes ``wikimedia/wikipedia`` and evaluates with
    ``sentence-transformers/natural-questions``. Natural Questions provides
    answers, but RAGPerf does not provide gold Wikipedia document or chunk IDs.
    """
    corpus_size = int(
        os.getenv("RAGPERF_WIKIPEDIA_CORPUS_SIZE", str(max(sample_size * 20, 1000)))
    )

    console.print(
        "[bold blue]Loading RAGPerf-style Wikipedia corpus "
        f"({RAGPERF_WIKIPEDIA_CONFIG}, {corpus_size} docs)...[/bold blue]"
    )
    wiki = load_dataset(
        RAGPERF_WIKIPEDIA_HF_ID, RAGPERF_WIKIPEDIA_CONFIG, split="train"
    )
    if corpus_size and corpus_size < len(wiki):
        wiki = wiki.shuffle(seed=42).select(range(corpus_size))

    corpus = [
        {
            "context": str(row.get("text", "")),
            "metadata": {
                "id": row.get("id"),
                "title": row.get("title"),
                "source_dataset": RAGPERF_WIKIPEDIA_HF_ID,
                "source_config": RAGPERF_WIKIPEDIA_CONFIG,
            },
        }
        for row in wiki
        if row.get("text")
    ]

    console.print(
        "[bold blue]Loading Natural Questions QA pairs "
        f"({sample_size} questions)...[/bold blue]"
    )
    nq = load_dataset(RAGPERF_NQ_HF_ID, split="train")
    if sample_size and sample_size < len(nq):
        nq = nq.shuffle(seed=42).select(range(sample_size))

    questions = normalize_samples(
        [
            {
                "question": row["query"],
                "ground_truth": str(row["answer"]),
                "context": "",
                "metadata": {
                    "source_dataset": RAGPERF_NQ_HF_ID,
                    "retrieval_ground_truth": "unavailable",
                },
            }
            for row in nq
        ],
        source=RAGPERF_NQ_HF_ID,
    )

    console.print(
        f"[green]Loaded {len(corpus)} Wikipedia documents and "
        f"{len(questions)} Natural Questions QA pairs[/green]"
    )
    return corpus, questions
