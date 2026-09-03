"""Tool suite for agentic RAG benchmarking.

Every tool is built by a factory function that closes over an injectable
backend, so tests can supply fakes and the adapter can wire real pipeline
components.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.tools import BaseTool

RetrieverFn = Callable[[str, int], list[Document]]
FilteredRetrieverFn = Callable[[str, int, dict | None], list[Document]]


def _format_documents(docs: list[Document]) -> str:
    parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"# Source: {source}\n\n{doc.page_content}")
        if index < len(docs):
            parts.append("\n\n")
    return "".join(parts).strip() or "No documents found."


def build_retrieve_tool(
    retriever: RetrieverFn, default_top_k: int = 4
) -> BaseTool:
    """Build the main vector-store retrieval tool."""

    @tool
    def retrieve(query: str, top_k: int = default_top_k) -> str:
        """Search the document corpus and return the most relevant chunks.

        Args:
            query: Natural language search query.
            top_k: Maximum number of chunks to return.
        """
        docs = retriever(query, top_k)
        return _format_documents(docs)

    return retrieve


def build_retrieve_with_filter_tool(
    retriever: FilteredRetrieverFn, default_top_k: int = 4
) -> BaseTool:
    """Build a metadata-filtered (self-query style) retrieval tool."""

    @tool
    def retrieve_with_filter(
        query: str, filters: str, top_k: int = default_top_k
    ) -> str:
        """Search the document corpus restricted to a metadata filter.

        Args:
            query: Natural language search query.
            filters: JSON object mapping metadata keys to values, e.g.
                '{"source": "report.md"}'.
            top_k: Maximum number of chunks to return.
        """
        try:
            parsed_filters: dict | None = json.loads(filters)
        except json.JSONDecodeError as exc:
            return f"Error: invalid filters JSON ({exc})."
        if parsed_filters is not None and not isinstance(parsed_filters, dict):
            return "Error: filters must be a JSON object."
        docs = retriever(query, top_k, parsed_filters)
        return _format_documents(docs)

    return retrieve_with_filter


def build_retrieve_alternate_tool(
    retriever: RetrieverFn, default_top_k: int = 4
) -> BaseTool:
    """Build a retrieval tool over the alternate collection for routing."""

    @tool
    def retrieve_alternate(query: str, top_k: int = default_top_k) -> str:
        """Search the alternate document collection.

        Use when the main corpus search did not yield useful results or the
        question may be better covered by the alternate index.

        Args:
            query: Natural language search query.
            top_k: Maximum number of chunks to return.
        """
        docs = retriever(query, top_k)
        return _format_documents(docs)

    return retrieve_alternate


def build_hyde_retrieve_tool(
    hypothesize: Callable[[str], str],
    retriever: RetrieverFn,
    default_top_k: int = 4,
) -> BaseTool:
    """Build a HyDE retrieval tool (search with a hypothetical answer)."""

    @tool
    def hyde_retrieve(query: str, top_k: int = default_top_k) -> str:
        """Search the corpus using HyDE: a hypothetical answer is generated
        first and used as the search query. Useful for vocabularly-mismatch
        questions.

        Args:
            query: Natural language search query.
            top_k: Maximum number of chunks to return.
        """
        hypothetical = hypothesize(query)
        docs = retriever(hypothetical, top_k)
        return _format_documents(docs)

    return hyde_retrieve


def build_rewrite_query_tool(llm: Callable[[str], str]) -> BaseTool:
    """Build an LLM-based query rewriting tool."""

    @tool
    def rewrite_query(query: str) -> str:
        """Rewrite a query to improve retrieval, resolving vague wording
        and adding likely search terms.

        Args:
            query: The original natural language query.
        """
        prompt = (
            "Rewrite the following search query to improve retrieval "
            "results. Keep the intent, resolve vague wording, and add "
            "likely domain terms. Return only the rewritten query.\n\n"
            f"Query: {query}"
        )
        return llm(prompt).strip()

    return rewrite_query


def build_rerank_chunks_tool(
    reranker: Callable[[str, list[Document], int], list[Document]],
) -> BaseTool:
    """Build a reranking tool over previously retrieved chunk texts."""

    @tool
    def rerank_chunks(query: str, documents: list[str], top_k: int = 4) -> str:
        """Re-rank retrieved chunks against the query and return the
        most relevant ones in order.

        Args:
            query: The query to score relevance against.
            documents: Chunk texts to re-rank.
            top_k: Number of top chunks to keep.
        """
        docs = [Document(page_content=text) for text in documents]
        ranked = reranker(query, docs, top_k)
        return "\n".join(doc.page_content for doc in ranked) or "No documents found."

    return rerank_chunks


def build_grade_relevance_tool(
    grader: Callable[[str, str], bool],
) -> BaseTool:
    """Build a tool that grades one document's relevance to a query."""

    @tool
    def grade_relevance(query: str, document: str) -> str:
        """Grade whether a retrieved chunk is relevant to the query.

        Args:
            query: The user query the chunk should answer.
            document: The chunk text to grade.
        """
        return "relevant" if grader(query, document) else "not relevant"

    return grade_relevance


def build_grade_answer_tool(
    checker: Callable[[str, list[str]], bool],
) -> BaseTool:
    """Build a self-check tool for answer faithfulness against contexts."""

    @tool
    def grade_answer(answer: str, contexts: list[str]) -> str:
        """Check whether the drafted answer is fully supported by the
        retrieved contexts.

        Args:
            answer: The drafted answer text.
            contexts: The retrieved chunk texts backing the answer.
        """
        return "supported" if checker(answer, contexts) else "unsupported"

    return grade_answer


def _default_web_search() -> Callable[[str], str]:
    def search(query: str) -> str:
        from langchain_community.tools import DuckDuckGoSearchRun

        return DuckDuckGoSearchRun().run(query)

    return search


def build_web_search_tool(
    search_fn: Callable[[str], str] | None = None,
) -> BaseTool:
    """Build a web search tool (DuckDuckGo by default)."""

    @tool
    def web_search(query: str) -> str:
        """Search the public web for information.

        Use as a fallback when the local corpus does not contain the answer.

        Args:
            query: The search query.
        """
        return (search_fn or _default_web_search())(query)

    return web_search


def _default_fetch_url() -> Callable[[str], str]:
    def fetch(url: str) -> str:
        import requests

        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text

    return fetch


def build_fetch_url_tool(
    fetch_fn: Callable[[str], str] | None = None,
    allowed_domains: list[str] | None = None,
) -> BaseTool:
    """Build a URL fetching tool, optionally restricted to domains."""

    @tool
    def fetch_url(url: str) -> str:
        """Fetch the text content of a web page.

        Args:
            url: The URL to fetch.
        """
        if allowed_domains and not any(
            url.startswith(domain) for domain in allowed_domains
        ):
            allowed = ", ".join(allowed_domains)
            return f"Error: URL not allowed. Must start with one of: {allowed}."
        return (fetch_fn or _default_fetch_url())(url)

    return fetch_url


def build_tool_suite(
    *,
    retriever: FilteredRetrieverFn,
    alternate_retriever: RetrieverFn | None = None,
    reranker: Callable[[str, list[Document], int], list[Document]] | None = None,
    llm: Callable[[str], str] | None = None,
    hypothesize: Callable[[str], str] | None = None,
    grader: Callable[[str, str], bool] | None = None,
    answer_checker: Callable[[str, list[str]], bool] | None = None,
    search_fn: Callable[[str], str] | None = None,
    fetch_fn: Callable[[str], str] | None = None,
    allowed_domains: list[str] | None = None,
    default_top_k: int = 4,
) -> list[BaseTool]:
    """Build the full agentic RAG tool suite.

    Tools are only included when their required backend is provided;
    `retrieve` and `retrieve_with_filter` always use `retriever`.
    """
    suite: list[BaseTool] = [
        build_retrieve_tool(retriever, default_top_k),
        build_retrieve_with_filter_tool(retriever, default_top_k),
    ]
    if alternate_retriever is not None:
        suite.append(
            build_retrieve_alternate_tool(alternate_retriever, default_top_k)
        )
    if hypothesize is not None:
        suite.append(
            build_hyde_retrieve_tool(hypothesize, retriever, default_top_k)
        )
    if llm is not None:
        suite.append(build_rewrite_query_tool(llm))
    if reranker is not None:
        suite.append(build_rerank_chunks_tool(reranker))
    if grader is not None:
        suite.append(build_grade_relevance_tool(grader))
    if answer_checker is not None:
        suite.append(build_grade_answer_tool(answer_checker))
    if search_fn is not None or fetch_fn is not None:
        suite.append(build_web_search_tool(search_fn))
    if fetch_fn is not None:
        suite.append(build_fetch_url_tool(fetch_fn, allowed_domains))
    return suite
