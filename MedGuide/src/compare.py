"""Report the rank of a known-correct chunk under each retrieval setup.

Example:
  python src/compare.py "What is the first-line medicine for type 2 diabetes?" \
      --gold-section 1.13.1 --gold-text "offer: modified-release metformin"

A chunk counts as 'gold' if its section contains --gold-section AND its text
contains --gold-text. The script prints the matched chunks so you can check
you picked the right one. Rank is the best rank among matches; 'not found'
means the chunk was not among the candidates that setup considered.
"""
import argparse
import hybrid


def best_rank(order, gold):
    ranks = [r for r, i in enumerate(order, 1) if i in gold]
    return min(ranks) if ranks else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--gold-section", required=True)
    ap.add_argument("--gold-text", required=True)
    a = ap.parse_args()

    chunks = hybrid.load()[0]
    gold = {i for i, c in enumerate(chunks)
            if a.gold_section in c["section"] and a.gold_text in c["text"]}
    print(f"matched {len(gold)} gold chunk(s):")
    for i in sorted(gold):
        print("  -", chunks[i]["section"], "|", chunks[i]["text"][:90])
    if not gold:
        raise SystemExit("No gold chunk matched: adjust --gold-section / --gold-text.")

    n = len(chunks)
    setups = [
         ("dense only", hybrid.search(a.question, "dense", n)),
        ("BM25 only", hybrid.search(a.question, "bm25", n)),
        ("hybrid (dense + BM25, RRF)", hybrid.search(a.question, "hybrid", n)),
        ("hybrid + rerank, pool 50", hybrid.search(a.question, "rerank", 50)),
        ("hybrid + rerank, pool 100", hybrid.search(a.question, "rerank", 100)),
        ("hybrid + rerank, pool 200", hybrid.search(a.question, "rerank", 200)),
        ("hybrid + rerank, pool 616 (all)", hybrid.search(a.question, "rerank", 616)),
    
    ]
    print(f"\nQuestion: {a.question}\n")
    for name, order in setups:
        r = best_rank(order, gold)
        print(f"{name:32s} rank of gold chunk: {r if r else 'not found'}")
