"""Download guideline pages listed in data/urls.txt (polite: 1 request / 2 s)."""
import hashlib, json, re, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "MedGuide-RAG-student-project (educational, non-commercial)"}


def slugify(url: str) -> str:
    s = re.sub(r"https?://(www\.)?", "", url).strip("/")
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s[:80] + "_" + hashlib.md5(url.encode()).hexdigest()[:6]


def main():
    urls = [l.strip() for l in (ROOT / "data" / "urls.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for url in urls:
        path = RAW / f"{slugify(url)}.html"
        if not path.exists():
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            path.write_text(r.text, encoding="utf-8")
            print("downloaded", url)
            time.sleep(2)
        else:
            print("cached    ", url)
        manifest.append({"url": url, "file": path.name})
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
