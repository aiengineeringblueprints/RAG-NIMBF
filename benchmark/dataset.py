from datasets import load_dataset
from rich.console import Console

console = Console()


def _build_context(row: dict) -> str:
    parts = []
    if row.get("pre_text"):
        parts.append(row["pre_text"])
    if row.get("table"):
        parts.append(row["table"])
    if row.get("post_text"):
        parts.append(row["post_text"])
    if not parts and row.get("context"):
        parts.append(row["context"])
    return "\n\n".join(parts)


def load_benchmark_data(subset: str = "FinQA", sample_size: int = 50) -> list[dict]:
    console.print(f"[bold blue]Loading T2-RAGBench ({subset})...[/bold blue]")
    ds = load_dataset("G4KMU/t2-ragbench", subset)

    split = "test" if "test" in ds else list(ds.keys())[0]
    data = ds[split]

    if sample_size and sample_size < len(data):
        data = data.shuffle(seed=42).select(range(sample_size))

    samples = []
    for row in data:
        samples.append({
            "question": row["question"],
            "ground_truth": str(row["program_answer"]),
            "context": _build_context(row),
            "metadata": {
                k: row.get(k)
                for k in [
                    "file_name", "company_name", "company_symbol",
                    "report_year", "page_number", "context_id",
                ]
                if k in row
            },
        })

    console.print(f"[green]Loaded {len(samples)} samples from {subset} ({split} split)[/green]")
    return samples
