"""Parse saved HTML into sections that keep their heading path.

Output: data/processed/sections.jsonl
  {doc_id, title, url, heading_path: [..], text}
"""
import json
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "sections.jsonl"
HEADINGS = ["h1", "h2", "h3", "h4"]


def clean(t: str) -> str:
    return " ".join(t.split())


def parse_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    root = soup.find("main") or soup.body or soup
    title = clean(soup.title.get_text()) if soup.title else "untitled"

    stack, buf, sections = [], [], []

    def flush():
        text = clean(" ".join(buf))
        if text:
            sections.append({"heading_path": list(stack), "text": text})
        buf.clear()

    for el in root.find_all(HEADINGS + ["p", "li"]):
        if el.name in ("p", "li") and el.find_parent("li"):
            continue  # already included in its parent <li>
        text = clean(el.get_text(" "))
        if not text:
            continue
        if el.name in HEADINGS:
            flush()
            level = HEADINGS.index(el.name)
            stack[:] = stack[:level] + [text]
        else:
            buf.append(text)
    flush()
    return title, sections


def main():
    manifest = json.loads((RAW / "manifest.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for item in manifest:
            html = (RAW / item["file"]).read_text(encoding="utf-8")
            title, sections = parse_html(html)
            for s in sections:
                rec = {"doc_id": item["file"], "title": title, "url": item["url"], **s}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} sections -> {OUT}")


if __name__ == "__main__":
    main()
