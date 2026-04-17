from datasets import load_dataset
from rich.console import Console

from benchmark.dataset_adapters import get_adapter

console = Console()


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
