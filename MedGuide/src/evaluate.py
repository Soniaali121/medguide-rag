"""Evaluate retrieval setups on eval/questions.jsonl.

Usage:
  python src/evaluate.py --no-rerank      # fast: dense, BM25, hybrid only
  python src/evaluate.py                  # also runs hybrid + reranker (slow on CPU)
  python src/evaluate.py --pool 100       # candidates given to the reranker

Metrics (answerable questions only; ranks beyond MAX_RANK count as misses):
  Recall@k : share of questions with at least ONE gold chunk in the top k
  MRR      : mean of 1 / (rank of the first gold chunk)
Results are also saved to eval/results.json for failure analysis.

Question format (one JSON object per line):
  {"id","question","gold":[{"section","text"}],"answerable":bool,"type":"direct|vague|unanswerable"}
A gold entry matches a chunk if its section path CONTAINS "section" and its text CONTAINS "text".
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import hybrid

ROOT = Path(__file__).resolve().parent.parent
QFILE = ROOT / "eval" / "questions.jsonl"
OUT = ROOT / "eval" / "results.json"
KS = (1, 3, 5, 10)
MAX_RANK = 100


def load_questions():
    with QFILE.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def gold_indices(q, chunks):
    gold = set()
    for g in q["gold"]:
        hits = {i for i, c in enumerate(chunks)
                if g["section"] in c["section"] and g["text"] in c["text"]}
        if not hits:
            print(f"WARNING {q['id']}: gold entry matches NO chunk -> {g}")
        gold |= hits
    return gold


def first_rank(order, gold):
    for r, i in enumerate(order, 1):
        if i in gold:
            return r if r <= MAX_RANK else None
    return None


def summarize(ranks):
    n = len(ranks)
    res = {f"R@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    res["MRR"] = sum(1 / r for r in ranks if r) / n
    return res


def top_dense_score(question):
    _, index, embedder, _ = hybrid.load()
    v = embedder.encode([hybrid.QUERY_PREFIX + question], normalize_embeddings=True)
    s, _ = index.search(np.asarray(v, dtype="float32"), 1)
    return float(s[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--pool", type=int, default=100)
    a = ap.parse_args()

    chunks = hybrid.load()[0]
    questions = load_questions()
    setups = {
        "dense only": lambda q: hybrid.search(q, "dense", MAX_RANK),
        "BM25 only": lambda q: hybrid.search(q, "bm25", MAX_RANK),
        "hybrid (RRF)": lambda q: hybrid.search(q, "hybrid", MAX_RANK),
    }
    if not a.no_rerank:
        setups[f"hybrid + rerank (pool {a.pool})"] = lambda q: hybrid.search(q, "rerank", a.pool)

    results = {}
    t0 = time.time()
    for n, q in enumerate(questions, 1):
        gold = gold_indices(q, chunks) if q["answerable"] else set()
        if q["answerable"] and not gold:
            print(f"WARNING {q['id']}: answerable but no gold chunk found -> skipped")
            continue
        rec = {"question": q["question"], "type": q["type"], "answerable": q["answerable"],
               "top_dense_score": top_dense_score(q["question"]), "ranks": {}}
        if q["answerable"]:
            for name, fn in setups.items():
                rec["ranks"][name] = first_rank(fn(q["question"]), gold)
        results[q["id"]] = rec
        print(f"[{n}/{len(questions)}] {q['id']} done ({time.time() - t0:.0f}s)", flush=True)

    answerable = {k: v for k, v in results.items() if v["answerable"]}
    if not answerable:
        raise SystemExit("No answerable questions with valid gold labels.")
    groups = {"all answerable": list(answerable)}
    for t in sorted({v["type"] for v in answerable.values()}):
        groups[f"type = {t}"] = [k for k, v in answerable.items() if v["type"] == t]

    for gname, ids in groups.items():
        print(f"\n### {gname} (n = {len(ids)})")
        print("| Setup | " + " | ".join(f"R@{k}" for k in KS) + " | MRR |")
        print("|---|" + "---|" * (len(KS) + 1))
        for name in setups:
            s = summarize([answerable[i]["ranks"][name] for i in ids])
            print(f"| {name} | " + " | ".join(f"{s[f'R@{k}']:.2f}" for k in KS) + f" | {s['MRR']:.3f} |")

    print("\n### Per-question rank of the first gold chunk (blank = not in top "
          f"{MAX_RANK})")
    print("| id | type | " + " | ".join(setups) + " |")
    print("|---|---|" + "---|" * len(setups))
    for i, v in answerable.items():
        print(f"| {i} | {v['type']} | " + " | ".join(str(v["ranks"][n] or "") for n in setups) + " |")

    ans = [v["top_dense_score"] for v in results.values() if v["answerable"]]
    una = [v["top_dense_score"] for v in results.values() if not v["answerable"]]
    if ans and una:
        print("\n### Top-1 dense similarity (for a future abstention threshold)")
        print(f"answerable   : mean {np.mean(ans):.3f}, min {min(ans):.3f}, max {max(ans):.3f}")
        print(f"unanswerable : mean {np.mean(una):.3f}, min {min(una):.3f}, max {max(una):.3f}")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
