"""Small stdio MCP server exposing lexical search over a local text corpus."""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

from mcp.server.fastmcp import FastMCP

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _load_corpus(root: Path) -> list[dict[str, str]]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Corpus path is not a directory: {resolved}")
    documents = []
    for path in sorted(resolved.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            documents.append(
                {
                    "source": str(path.relative_to(resolved)),
                    "text": path.read_text(encoding="utf-8"),
                }
            )
    if not documents:
        raise ValueError(f"No .md or .txt documents found below {resolved}")
    return documents


def _search(
    documents: list[dict[str, str]], query: str, top_k: int
) -> list[dict[str, object]]:
    query_terms = Counter(_tokens(query))
    if not query_terms:
        return []

    document_terms = [Counter(_tokens(item["text"])) for item in documents]
    document_frequency = {
        term: sum(term in terms for terms in document_terms) for term in query_terms
    }
    scored: list[tuple[float, dict[str, str]]] = []
    for document, terms in zip(documents, document_terms):
        score = 0.0
        for term, query_frequency in query_terms.items():
            if terms[term]:
                inverse_document_frequency = (
                    math.log((len(documents) + 1) / (document_frequency[term] + 1))
                    + 1.0
                )
                score += (
                    query_frequency
                    * (1.0 + math.log(terms[term]))
                    * inverse_document_frequency
                )
        if score > 0:
            scored.append((score, document))

    scored.sort(key=lambda item: (-item[0], item[1]["source"]))
    return [
        {"text": document["text"], "source": document["source"], "score": score}
        for score, document in scored[: max(1, top_k)]
    ]


def _best_sentence(text: str, query: str) -> str:
    query_terms = set(_tokens(query))
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip() and not sentence.lstrip().startswith("#")
    ]
    if not sentences:
        return ""
    return max(
        sentences,
        key=lambda sentence: (
            len(query_terms.intersection(_tokens(sentence))),
            -len(sentence),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    args = parser.parse_args()
    documents = _load_corpus(args.corpus)
    documents_by_source = {item["source"]: item for item in documents}
    server = FastMCP("benchmark-corpus-search")

    @server.tool()
    def search(query: str, top_k: int = 5) -> dict[str, object]:
        """Return the most relevant corpus documents for a question."""
        return {"contexts": _search(documents, query, top_k)}

    @server.tool()
    def lookup(source_id: str) -> dict[str, object]:
        """Read one corpus document by its source ID or filename."""
        if not SOURCE_ID_PATTERN.fullmatch(source_id):
            raise ValueError("source_id contains unsupported characters")
        candidates = (source_id, f"{source_id}.md", f"{source_id}.txt")
        document = next(
            (
                documents_by_source[name]
                for name in candidates
                if name in documents_by_source
            ),
            None,
        )
        if document is None:
            raise ValueError(f"Unknown source_id: {source_id}")
        return {"contexts": [document]}

    @server.tool()
    def answer(query: str) -> dict[str, object]:
        """Return a lexical extractive answer for an MCP-only QA baseline."""
        matches = _search(documents, query, 1)
        if not matches:
            return {"answer": "", "source": None}
        match = matches[0]
        return {
            "answer": _best_sentence(str(match["text"]), query),
            "source": match["source"],
        }

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
