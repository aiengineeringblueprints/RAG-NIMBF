import os

from datasets import load_dataset
from rich.console import Console

from benchmark.dataset_adapters import get_adapter

console = Console()

RAGPERF_WIKIPEDIA_DATASET = "ragperf-wikipedia-nq"
RAGPERF_WIKIPEDIA_HF_ID = "wikimedia/wikipedia"
RAGPERF_WIKIPEDIA_CONFIG = "20231101.en"
RAGPERF_NQ_HF_ID = "sentence-transformers/natural-questions"


def load_benchmark_data(
    dataset_name: str = "t2-ragbench",
    subset: str | None = None,
    sample_size: int = 50,
) -> list[dict]:
    adapter = get_adapter(dataset_name)

    label = subset or "default"
    console.print(f"[bold blue]Loading {adapter.hf_id} ({label})...[/bold blue]")

    kwargs: dict = {}
    if adapter.requires_subset and subset:
        kwargs["name"] = subset
    ds = load_dataset(adapter.hf_id, **kwargs)

    split = adapter.preferred_split
    if split not in ds:
        split = list(ds.keys())[0]
    data = ds[split]

    if sample_size and sample_size < len(data):
        data = data.shuffle(seed=42).select(range(sample_size))

    samples = []
    for row in data:
        gt_raw = row.get(adapter.ground_truth_key, "")
        if adapter.ground_truth_transform:
            gt = adapter.ground_truth_transform(gt_raw)
        else:
            gt = str(gt_raw)

        samples.append({
            "question": row[adapter.question_key],
            "ground_truth": gt,
            "context": adapter.build_context(row),
            "metadata": {
                k: row.get(k)
                for k in adapter.metadata_keys
                if k in row
            },
        })

    console.print(
        f"[green]Loaded {len(samples)} samples from {adapter.hf_id} "
        f"({split} split)[/green]"
    )
    return samples


def load_corpus_and_questions(
    dataset_name: str = "squad",
    subset: str | None = None,
    sample_size: int = 50,
) -> tuple[list[dict], list[dict]]:
    """Load data and split into a deduplicated corpus and per-question entries.

    Returns (corpus, questions) where:
      - corpus: list of {context, metadata} dicts — all unique contexts
      - questions: list of {question, ground_truth, context, metadata} dicts
    """
    if dataset_name == RAGPERF_WIKIPEDIA_DATASET:
        return _load_ragperf_wikipedia_nq(sample_size=sample_size)

    samples = load_benchmark_data(dataset_name, subset, sample_size)

    seen: dict[str, int] = {}  # context text → index in corpus
    corpus: list[dict] = []

    for sample in samples:
        ctx = sample["context"]
        if ctx not in seen:
            seen[ctx] = len(corpus)
            corpus.append({
                "context": ctx,
                "metadata": sample.get("metadata", {}),
            })

    console.print(
        f"[green]Deduplicated corpus: {len(corpus)} unique documents "
        f"for {len(samples)} questions[/green]"
    )
    return corpus, samples


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
    wiki = load_dataset(RAGPERF_WIKIPEDIA_HF_ID, RAGPERF_WIKIPEDIA_CONFIG, split="train")
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

    questions = [
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
    ]

    console.print(
        f"[green]Loaded {len(corpus)} Wikipedia documents and "
        f"{len(questions)} Natural Questions QA pairs[/green]"
    )
    return corpus, questions
