import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, cast

from datasets import load_dataset

from rlm import RLM
from rlm.clients import get_client
from rlm.logger.rlm_logger import RLMLogger
from rlm.utils.dynamic_model_picker_prompt import (
    RLM_SYSTEM_PROMPT as DYNAMIC_MODEL_PICKER_PROMPT,
)
from rlm.utils.subagent_confidence_selfeval_prompt import (
    RLM_SYSTEM_PROMPT as SUBAGENT_CONFIDENCE_SELFEVAL_PROMPT,
)
from rlm.utils.subagent_encouraging_prompt import (
    RLM_SYSTEM_PROMPT as SUBAGENT_ENCOURAGING_PROMPT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decrypted_path",
        default="bench_BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
    )
    parser.add_argument(
        "--corpus_path",
        default="bench_BrowseComp-Plus/corpus.jsonl",
    )
    parser.add_argument("--num_docs", type=int, default=1000)
    parser.add_argument("--num_docs_test", type=int, default=5)
    parser.add_argument("--smoke_test", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--query_index",
        type=int,
        default=0,
        help="0-indexed record offset into the decrypted JSONL.",
    )
    group.add_argument(
        "--query_id",
        default=None,
        help="Select the record with matching query_id (string match).",
    )
    parser.add_argument(
        "--model_name",
        default=os.getenv("BROWSECOMP_MODEL", "gpt-5.4-mini"),
    )
    parser.add_argument(
        "--system_prompt",
        choices=[
            "default",
            "subagent_encouraging",
            "subagent_confidence_selfeval",
            "dynamic_model_picker",
        ],
        default="default",
    )
    parser.add_argument(
        "--mode",
        choices=["rlm", "default"],
        default="rlm",
    )
    return parser.parse_args()


def ensure_decrypted_dataset(decrypted_path: str) -> None:
    if os.path.exists(decrypted_path):
        return

    raise FileNotFoundError(
        f"Missing decrypted dataset at '{decrypted_path}'. "
        "Run: python bench_BrowseComp-Plus/decrypt_dataset.py"
    )


def write_corpus_jsonl(corpus_path: str, limit: int | None) -> None:
    os.makedirs(os.path.dirname(corpus_path) or ".", exist_ok=True)

    ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            json.dump(dict(row), f, ensure_ascii=False)
            f.write("\n")


def count_jsonl_records(path: str, *, max_count: int) -> int:
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            if count >= max_count:
                return count
    return count


def ensure_corpus(
    corpus_path: str,
    *,
    smoke_test: bool,
    smoke_test_limit: int,
    min_docs_required: int,
) -> None:
    if min_docs_required <= 0:
        raise ValueError("min_docs_required must be > 0")

    if os.path.exists(corpus_path):
        existing = count_jsonl_records(corpus_path, max_count=min_docs_required)
        if existing >= min_docs_required:
            return

    limit = smoke_test_limit if smoke_test else None
    write_corpus_jsonl(corpus_path, limit=limit)


def load_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def load_query_record(
    decrypted_path: str, *, query_index: int, query_id: str | None
) -> dict[str, Any]:
    if query_id is None and query_index < 0:
        raise ValueError("query_index must be >= 0")

    sample_query_ids: list[str] = []
    with open(decrypted_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row_query_id = str(row.get("query_id", ""))
            if row_query_id and len(sample_query_ids) < 5:
                sample_query_ids.append(row_query_id)
            if query_id is not None:
                if row_query_id == query_id:
                    return row
            else:
                if i == query_index:
                    return row

    if query_id is not None:
        sample_suffix = (
            f" Example query_id values in this file: {', '.join(sample_query_ids)}."
            if sample_query_ids
            else ""
        )
        if query_id.isdigit():
            raise ValueError(
                f"No record found with query_id='{query_id}' in '{decrypted_path}'. "
                f"If you meant the 0-indexed row offset, use --query_index {query_id} instead."
                f"{sample_suffix}"
            )
        raise ValueError(
            f"No record found with query_id='{query_id}' in '{decrypted_path}'.{sample_suffix}"
        )
    raise ValueError(f"No record found at query_index={query_index} in '{decrypted_path}'")


@dataclass(frozen=True)
class CorpusDoc:
    docid: str
    title: str
    text: str


def _doc_from_record(d: dict[str, Any]) -> CorpusDoc:
    return CorpusDoc(
        docid=str(d.get("docid", "")),
        title=str(d.get("title", "")),
        text=str(d.get("text", "")),
    )


def build_smoke_test_docs(
    query_record: dict[str, Any], *, total_docs: int, seed: int = 0
) -> list[CorpusDoc]:
    gold_docs_raw = query_record.get("gold_docs")
    if not isinstance(gold_docs_raw, list) or not gold_docs_raw:
        raise ValueError("Expected non-empty list field `gold_docs` in decrypted record")

    gold_doc = _doc_from_record(gold_docs_raw[0])
    if not gold_doc.docid:
        raise ValueError("Gold doc missing `docid` in decrypted record")

    candidates_raw: list[dict[str, Any]] = []
    for key in ("negative_docs", "evidence_docs"):
        v = query_record.get(key)
        if isinstance(v, list):
            candidates_raw.extend([x for x in v if isinstance(x, dict)])

    candidates = [_doc_from_record(d) for d in candidates_raw]
    candidates = [d for d in candidates if d.docid and d.docid != gold_doc.docid]
    if len(candidates) < (total_docs - 1):
        raise ValueError(
            f"Not enough non-gold candidates to sample {total_docs - 1} docs; got {len(candidates)}"
        )

    rng = random.Random(seed)
    sampled = rng.sample(candidates, k=total_docs - 1)
    docs = [gold_doc, *sampled]
    rng.shuffle(docs)
    return docs


def ensure_gold_doc_in_context(
    corpus_docs: list[CorpusDoc], *, gold_doc: CorpusDoc, total_docs: int
) -> tuple[list[CorpusDoc], bool]:
    if total_docs <= 0:
        raise ValueError("total_docs must be > 0")
    if not gold_doc.docid:
        raise ValueError("gold_doc must have non-empty docid")

    if len(corpus_docs) < total_docs:
        raise ValueError(
            f"corpus_docs has {len(corpus_docs)} docs but total_docs={total_docs}; "
            "load a larger slice from corpus.jsonl before injecting gold"
        )

    # Deduplicate by docid while preserving order.
    seen: set[str] = set()
    unique: list[CorpusDoc] = []
    for d in corpus_docs:
        if not d.docid:
            continue
        if d.docid in seen:
            continue
        seen.add(d.docid)
        unique.append(d)

    if len(unique) < total_docs:
        raise ValueError(
            f"corpus_docs collapses to {len(unique)} unique docids but total_docs={total_docs}; "
            "cannot build a full context without duplicates"
        )

    gold_docid = gold_doc.docid
    injected = False

    # Start with the first total_docs docs.
    selected = unique[:total_docs]
    selected_docids = {d.docid for d in selected}

    if gold_docid in selected_docids:
        return selected, injected

    # Gold doc exists in the slice but beyond the cutoff: swap it in.
    gold_in_unique = next((d for d in unique if d.docid == gold_docid), None)
    selected[-1] = gold_in_unique if gold_in_unique is not None else gold_doc
    injected = True

    # Final invariant.
    if len(selected) != total_docs:
        raise RuntimeError("Failed to construct corpus slice with the requested total_docs")
    if gold_docid not in {d.docid for d in selected}:
        raise RuntimeError("Failed to include gold docid in context")

    return selected, injected


def normalize_corpus_docs(raw_docs: list[dict[str, Any]]) -> list[CorpusDoc]:
    out: list[CorpusDoc] = []
    for d in raw_docs:
        docid = str(d.get("docid", ""))
        title = str(d.get("title", ""))
        text = str(d.get("text", ""))
        out.append(CorpusDoc(docid=docid, title=title, text=text))
    return out


def build_context_payload(
    query_record: dict[str, Any], corpus_docs: list[CorpusDoc]
) -> dict[str, Any]:
    return {
        "query": query_record.get("query"),
        "query_id": query_record.get("query_id"),
        "answers": query_record.get("answers"),
        "corpus": [doc.__dict__ for doc in corpus_docs],
    }


def build_prompt() -> str:
    return (
        "You are running inside an RLM REPL. The variable `context` is available and contains: "
        "(1) a BrowseComp-Plus query in context['query'] and (2) a small corpus slice in context['corpus'] "
        "(list of {docid,title,text}).\n\n"
        "Task: Answer the query using ONLY the provided corpus slice. This is a multi-hop question; "
        "you may need to find an entity in one doc and then look it up in another doc.\n\n"
        "You should only do 1 thing per iteration. In the first iteration, inspect the query and corpus structure and narrow the corpus to a small set of candidate docs or excerpts. Wait for the actual result before taking the next step.\n\n"
        "Use Python only for mechanical work such as printing keys, scanning strings, filtering docs, extracting snippets, and slicing down the evidence. Do not use Python logic alone to decide which evidence answers the question, resolve ambiguity, compare competing candidates, or verify a semantic claim.\n\n"
        "Once you have a small candidate set, use `rlm_query` or `rlm_query_batched` for the semantic work: identifying the relevant entity, checking whether a document actually supports the claim, comparing candidates, and deciding which candidate is correct. Do not combine broad search, semantic reasoning, and final answer production in the same iteration.\n\n"
        "Return the final answer with FINAL_VAR('answer')."
    )


def build_baseline_prompt(
    query_record: dict[str, Any], corpus_docs: list[CorpusDoc]
) -> list[dict[str, str]]:
    corpus_text = "\n\n".join(
        (f"DocID: {doc.docid}\nTitle: {doc.title}\nText:\n{doc.text}") for doc in corpus_docs
    )
    return [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the provided BrowseComp-Plus corpus slice. "
                "Do not use external knowledge. Return only the final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {query_record.get('query')}\n\n"
                f"Corpus:\n{corpus_text}\n\n"
                "Return only the final answer."
            ),
        },
    ]


def get_custom_system_prompt(system_prompt: str) -> str | None:
    if system_prompt == "default":
        return None
    if system_prompt == "subagent_encouraging":
        return SUBAGENT_ENCOURAGING_PROMPT
    if system_prompt == "subagent_confidence_selfeval":
        return SUBAGENT_CONFIDENCE_SELFEVAL_PROMPT
    if system_prompt == "dynamic_model_picker":
        return DYNAMIC_MODEL_PICKER_PROMPT
    raise ValueError(f"Unsupported system_prompt='{system_prompt}'")


def get_backend_kwargs(model_name: str) -> dict[str, Any]:
    return {
        "api_key": os.getenv("OPENAI_API_KEY2"),
        "model_name": model_name,
    }


def main() -> None:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY2"):
        raise ValueError("OPENAI_API_KEY2 is required to run this benchmark runner.")

    ensure_decrypted_dataset(args.decrypted_path)

    num_docs = args.num_docs_test if args.smoke_test else args.num_docs
    query_record = load_query_record(
        args.decrypted_path,
        query_index=args.query_index,
        query_id=args.query_id,
    )

    if args.smoke_test:
        corpus_docs = build_smoke_test_docs(query_record, total_docs=num_docs)
        gold_docid = str(query_record["gold_docs"][0].get("docid", ""))
        selected_docids = [d.docid for d in corpus_docs]
        if gold_docid not in selected_docids:
            raise RuntimeError(
                "Smoke test invariant failed: gold docid not in selected corpus docs"
            )
        print("Smoke test gold docid:", gold_docid)
        print("Smoke test selected docids:", selected_docids)
    else:
        ensure_corpus(
            args.corpus_path,
            smoke_test=args.smoke_test,
            smoke_test_limit=args.num_docs_test,
            min_docs_required=num_docs,
        )
        raw_docs = load_jsonl(args.corpus_path, limit=num_docs)
        corpus_docs = normalize_corpus_docs(raw_docs)
        gold_docs_raw = query_record.get("gold_docs")
        if not isinstance(gold_docs_raw, list) or not gold_docs_raw:
            raise ValueError("Expected non-empty list field `gold_docs` in decrypted record")
        gold_doc = _doc_from_record(gold_docs_raw[0])
        corpus_docs, injected = ensure_gold_doc_in_context(
            corpus_docs,
            gold_doc=gold_doc,
            total_docs=num_docs,
        )
        if injected:
            print("Non-smoke: ensured gold docid is included in context:", gold_doc.docid)

    context_payload = build_context_payload(query_record, corpus_docs)
    backend_kwargs = get_backend_kwargs(args.model_name)
    subagent_backend_kwargs = get_backend_kwargs("gpt-5.4-mini")

    logger: RLMLogger | None = None
    rlm: RLM | None = None
    root_client = None

    if args.mode == "default":
        root_client = cast(
            Any,
            get_client("openai", backend_kwargs),
        )
    else:
        logger = RLMLogger(log_dir="./bench_BrowseComp-Plus/logs")
        rlm = RLM(
            backend="openai",
            backend_kwargs=backend_kwargs,
            subagent_backend_kwargs=subagent_backend_kwargs,
            environment="local",
            max_depth=2,
            compaction=True,
            verbose=True,
            logger=logger,
            custom_system_prompt=get_custom_system_prompt(args.system_prompt),
        )

    print("mode:", args.mode)
    print("model_name:", args.model_name)
    print("system_prompt:", args.system_prompt)

    if args.mode == "default":
        response = root_client.completion(build_baseline_prompt(query_record, corpus_docs))
    else:
        result = rlm.completion(context_payload, root_prompt=build_prompt())
        response = result.response

    print("Response:", response)
    print("Ground truth query_id:", query_record.get("query_id"))
    print("Ground truth answer:", query_record.get("answer"))
    if logger is not None and logger.log_file_path:
        print("Log file:", logger.log_file_path)
    if rlm is not None:
        rlm.close()


if __name__ == "__main__":
    main()
