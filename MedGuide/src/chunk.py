"""Section-aware chunking.

Each chunk stays inside one section and gets its heading path prepended,
so the embedding knows the context. Long sections are split on sentence
boundaries with a small overlap.

Output: data/processed/chunks.jsonl
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "sections.jsonl"
OUT = ROOT / "data" / "processed" / "chunks.jsonl"

MAX_WORDS = 220
OVERLAP_WORDS = 30
MIN_WORDS = 80  # Merge tiny sibling bullets until they reach at least 80 words


def split_sentences(text: str):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def split_section(text: str, max_words=MAX_WORDS, overlap=OVERLAP_WORDS):
    sents, chunks, cur, cur_len = split_sentences(text), [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and cur_len + w > max_words:
            chunks.append(" ".join(cur))
            tail, tl = [], 0
            for prev in reversed(cur):
                tl += len(prev.split())
                tail.insert(0, prev)
                if tl >= overlap:
                    break
            cur, cur_len = tail, sum(len(x.split()) for x in tail)
        cur.append(s)
        cur_len += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def merge_small_sections(sections: list[dict]) -> list[dict]:
    """
    Merges tiny consecutive subsections that share the same parent heading
    (e.g., 1.13.1, 1.13.2, 1.13.3 under 1.13) so chunks have full clinical context.
    """
    merged = []
    for sec in sections:
        path = sec["heading_path"]
        parent = tuple(path[:-1]) if len(path) > 1 else tuple(path)
        leaf = path[-1] if path else ""
        labeled_text = f"[{leaf}] {sec['text']}" if leaf else sec["text"]

        if merged:
            prev = merged[-1]
            prev_words = len(prev["text"].split())
            curr_words = len(labeled_text.split())

            # Merge if same document, same parent section, and combined size fits cleanly
            if (
                prev["doc_id"] == sec["doc_id"]
                and prev["_parent"] == parent
                and (prev_words < MIN_WORDS or curr_words < MIN_WORDS)
                and (prev_words + curr_words <= MAX_WORDS)
            ):
                prev["text"] += "\n" + labeled_text
                # Keep parent heading path so all merged sub-numbers match gold sections
                prev["heading_path"] = list(parent)
                continue

        merged.append({
            "doc_id": sec["doc_id"],
            "title": sec["title"],
            "url": sec["url"],
            "heading_path": path,
            "text": labeled_text,
            "_parent": parent,
        })
    return merged


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with IN.open(encoding="utf-8") as fin:
        raw_sections = [json.loads(line) for line in fin if line.strip()]

    sections = merge_small_sections(raw_sections)

    n = 0
    with OUT.open("w", encoding="utf-8") as fout:
        for sec in sections:
            path = " > ".join(sec["heading_path"]) or sec["title"]
            for piece in split_section(sec["text"]):
                rec = {
                    "chunk_id": f"{sec['doc_id']}::{n}",
                    "doc_id": sec["doc_id"],
                    "title": sec["title"],
                    "url": sec["url"],
                    "section": path,
                    "text": piece,
                    "embed_text": f"Guideline: {sec['title']}\nSection: {path}\n{piece}",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} chunks -> {OUT}")


if __name__ == "__main__":
    main()