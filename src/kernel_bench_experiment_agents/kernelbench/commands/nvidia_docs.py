"""Search and fetch official NVIDIA documentation through a narrow brokered tool."""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from kernel_bench_experiment_agents.runtime.common import emit_json


ALLOWED_DOCS_HOST = "docs.nvidia.com"
DEFAULT_INDEX_URLS = (
    "https://docs.nvidia.com/llms.txt",
    "https://docs.nvidia.com/cuda/llms.txt",
)
DEFAULT_MAX_RESULTS = 8
DEFAULT_MAX_CHARS = 12000
FETCH_TIMEOUT_SECONDS = 12
MAX_FETCH_BYTES = 1_000_000
USER_AGENT = "kernel-harness-nvidia-docs-broker/1.0"
QUERY_SYNONYMS = {
    "bf16": ("bfloat16", "bfloat", "tensor"),
    "bfloat16": ("bf16", "bfloat", "tensor"),
    "h100": ("hopper",),
    "wgmma": ("mma", "warpgroup", "hopper"),
    "wmma": ("mma", "matrix", "tensor"),
    "tensor": ("matrix", "mma"),
    "tensorcore": ("tensor", "mma"),
    "tensorcores": ("tensor", "mma"),
    "profile": ("nsight", "ncu"),
    "profiling": ("nsight", "ncu"),
}


@dataclass(frozen=True)
class DocsFetch:
    url: str
    text: str
    content_type: str
    truncated: bool


@dataclass(frozen=True)
class DocsLink:
    title: str
    url: str
    context: str
    source_url: str


PRIORITY_DOC_LINKS = (
    DocsLink(
        title="CUDA C++ Programming Guide",
        url="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html",
        context="Core CUDA programming model, memory hierarchy, WMMA, and CUDA C++ device programming reference.",
        source_url="builtin",
    ),
    DocsLink(
        title="PTX ISA Reference",
        url="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html",
        context="PTX instructions, inline assembly, mma/wgmma, tensor operations, and low-level GPU ISA behavior.",
        source_url="builtin",
    ),
    DocsLink(
        title="Hopper Tuning Guide",
        url="https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html.md",
        context="H100/Hopper CUDA kernel tuning, occupancy, memory hierarchy, tensor cores, and architecture-specific performance guidance.",
        source_url="builtin",
    ),
    DocsLink(
        title="Nsight Compute CLI Reference",
        url="https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html",
        context="Nsight Compute command-line profiling, metric collection, reports, and profiler options.",
        source_url="builtin",
    ),
    DocsLink(
        title="Nsight Compute Profiling Guide",
        url="https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html",
        context="Nsight Compute profiling workflow, metrics, bottleneck analysis, and kernel performance interpretation.",
        source_url="builtin",
    ),
)


def _validate_docs_url(raw_url: str, *, base_url: str | None = None) -> str:
    raw_url = str(raw_url or "").strip()
    if not raw_url:
        raise RuntimeError("NVIDIA docs URL is required")
    parsed = urlparse(urljoin(base_url or "", raw_url))
    if parsed.scheme != "https":
        raise RuntimeError("NVIDIA docs URL must use https")
    if parsed.hostname != ALLOWED_DOCS_HOST:
        raise RuntimeError(f"NVIDIA docs URL must be hosted on {ALLOWED_DOCS_HOST}")
    if parsed.username or parsed.password:
        raise RuntimeError("NVIDIA docs URL must not include credentials")
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path or "/",
            "",
            parsed.query,
            parsed.fragment,
        )
    )


def _decode_response(data: bytes, content_type: str) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([^;]+)", content_type, flags=re.IGNORECASE)
    if match:
        charset = match.group(1).strip()
    return data.decode(charset, errors="replace")


def _html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|section|article|li|tr|h[1-6])>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    return _normalize_text(text)


def _normalize_text(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        compact = re.sub(r"[ \t]+", " ", line).strip()
        if compact:
            lines.append(compact)
    return "\n".join(lines)


def _fetch_docs_text(url: str) -> DocsFetch:
    url = _validate_docs_url(url)
    request = Request(
        url,
        headers={
            "Accept": "text/plain,text/markdown,text/html,application/xhtml+xml",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            final_url = _validate_docs_url(response.geturl())
            content_type = response.headers.get("content-type", "")
            data = response.read(MAX_FETCH_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"NVIDIA docs fetch failed with HTTP {exc.code}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"NVIDIA docs fetch failed: {exc.reason}") from exc

    truncated = len(data) > MAX_FETCH_BYTES
    if truncated:
        data = data[:MAX_FETCH_BYTES]
    text = _decode_response(data, content_type)
    if "html" in content_type.lower() or "<html" in text[:512].lower():
        text = _html_to_text(text)
    else:
        text = _normalize_text(text)
    return DocsFetch(url=final_url, text=text, content_type=content_type, truncated=truncated)


def _tokens(value: str) -> list[str]:
    stop_words = {
        "and",
        "for",
        "from",
        "how",
        "the",
        "use",
        "with",
        "what",
        "when",
        "where",
    }
    tokens = [
        token
        for token in re.findall(r"[a-z0-9_+.:-]+", value.lower())
        if len(token) >= 2 and token not in stop_words
    ]
    expanded: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for candidate in (token, *QUERY_SYNONYMS.get(token, ())):
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def _extract_links(index: DocsFetch) -> list[DocsLink]:
    links: list[DocsLink] = []
    for line in index.text.splitlines():
        context = line.strip()
        markdown_urls: set[str] = set()
        for title, raw_url in re.findall(r"\[([^\]]+)\]\s*\(([^)]+)\)", line):
            try:
                url = _validate_docs_url(raw_url, base_url=index.url)
            except RuntimeError:
                continue
            markdown_urls.add(url)
            links.append(
                DocsLink(
                    title=_normalize_text(title) or url,
                    url=url,
                    context=context,
                    source_url=index.url,
                )
            )
        for raw_url in re.findall(r"https://docs\.nvidia\.com/[^\s)>\"]+", line):
            try:
                url = _validate_docs_url(raw_url)
            except RuntimeError:
                continue
            if url in markdown_urls:
                continue
            title = context.replace(raw_url, "").strip(" -:\t") or url
            links.append(
                DocsLink(
                    title=title,
                    url=url,
                    context=context,
                    source_url=index.url,
                )
            )
    return links


def _contains_term(haystack: str, token: str) -> bool:
    if len(token) <= 3:
        return re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", haystack) is not None
    return token in haystack


def _score_link(link: DocsLink, query_tokens: list[str]) -> int:
    haystacks = {
        "title": link.title.lower(),
        "url": link.url.lower(),
        "context": link.context.lower(),
    }
    score = 0
    for token in query_tokens:
        if _contains_term(haystacks["title"], token):
            score += 5
        if _contains_term(haystacks["url"], token):
            score += 3
        if _contains_term(haystacks["context"], token):
            score += 1
    return score


def _search_indexes(query: str, *, max_results: int) -> tuple[list[dict[str, Any]], list[str]]:
    query_tokens = _tokens(query)
    if not query_tokens:
        raise RuntimeError("query must contain at least one searchable term")

    errors: list[str] = []
    links_by_url: dict[str, DocsLink] = {}
    for link in PRIORITY_DOC_LINKS:
        links_by_url[link.url] = link
    for index_url in DEFAULT_INDEX_URLS:
        try:
            index = _fetch_docs_text(index_url)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        for link in _extract_links(index):
            links_by_url.setdefault(link.url, link)

    ranked: list[tuple[int, DocsLink]] = []
    for link in links_by_url.values():
        score = _score_link(link, query_tokens)
        if score > 0:
            ranked.append((score, link))
    ranked.sort(key=lambda item: (-item[0], item[1].title.lower(), item[1].url))

    if not ranked:
        fallback = sorted(links_by_url.values(), key=lambda item: (item.title.lower(), item.url))
        ranked = [(0, link) for link in fallback[:max_results]]

    results = [
        {
            "title": link.title,
            "url": link.url,
            "score": score,
            "source_url": link.source_url,
            "excerpt": link.context,
        }
        for score, link in ranked[:max_results]
    ]
    return results, errors


def _matching_excerpts(text: str, query: str, *, max_excerpts: int = 8) -> list[str]:
    tokens = _tokens(query)
    if not tokens:
        return []
    excerpts: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        lower = line.lower()
        if not any(_contains_term(lower, token) for token in tokens):
            continue
        start = max(0, index - 1)
        end = min(len(lines), index + 2)
        excerpt = "\n".join(lines[start:end])
        excerpts.append(excerpt)
        if len(excerpts) >= max_excerpts:
            break
    return excerpts


def _fetch_document(url: str, *, query: str, max_chars: int) -> dict[str, Any]:
    fetched = _fetch_docs_text(url)
    text = fetched.text
    clipped = text[:max_chars]
    return {
        "url": fetched.url,
        "content_type": fetched.content_type,
        "content": clipped,
        "content_chars": len(clipped),
        "source_chars": len(text),
        "content_truncated": fetched.truncated or len(text) > len(clipped),
        "matched_excerpts": _matching_excerpts(text, query),
    }


def command_research_nvidia_docs(args: argparse.Namespace) -> None:
    query = str(getattr(args, "query", "") or "").strip()
    url = str(getattr(args, "url", "") or "").strip()
    max_results = max(1, min(int(getattr(args, "max_results", DEFAULT_MAX_RESULTS)), 20))
    max_chars = max(1000, min(int(getattr(args, "max_chars", DEFAULT_MAX_CHARS)), 50000))
    if not query and not url:
        raise SystemExit("research_nvidia_docs requires --query, --url, or both")

    payload: dict[str, Any] = {
        "status": "succeeded",
        "tool": "research_nvidia_docs",
        "allowed_domains": [ALLOWED_DOCS_HOST],
        "query": query or None,
        "url": url or None,
        "max_results": max_results,
        "max_chars": max_chars,
        "results": [],
        "document": None,
        "errors": [],
    }
    if query:
        results, errors = _search_indexes(query, max_results=max_results)
        payload["results"] = results
        payload["errors"] = errors
    if url:
        payload["document"] = _fetch_document(url, query=query, max_chars=max_chars)
    emit_json(payload)


def dumps_for_tests(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
