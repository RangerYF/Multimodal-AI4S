"""Download the published Zenodo archives in verified byte ranges."""
import hashlib
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / "data/author"
FILES = {"training.zip": "19f79fd9746d9979b9768db99bd260a8",
         "user_study.zip": "2839967cf69090bb20520af4da0cbcd8"}


def download(name, expected):
    url = f"https://zenodo.org/records/16209653/files/{name}?download=1"
    target = ROOT / name
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == expected:
        print(f"Already verified: {name}", flush=True)
        return
    probe = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=45)
    probe.raise_for_status()
    size = int(probe.headers["Content-Range"].split("/")[-1])
    parts = ROOT / (name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    chunk = 4 * 1024 * 1024

    def fetch(start):
        end = min(start + chunk, size) - 1
        part = parts / str(start)
        if part.exists() and part.stat().st_size == end-start+1:
            return part
        for attempt in range(8):
            try:
                pending = part.with_suffix(".part")
                headers = part.with_suffix(".headers")
                subprocess.run(["curl.exe", "-L", "--fail", "--silent", "--show-error",
                                "--connect-timeout", "15", "--max-time", "90",
                                "--range", f"{start}-{end}", "-D", str(headers),
                                url, "-o", str(pending)], check=True)
                if f"content-range: bytes {start}-{end}/{size}" not in headers.read_text().lower() or pending.stat().st_size != end-start+1:
                    raise ValueError("Unexpected range response")
                pending.replace(part)
                print(f"{name}: {end+1}/{size}", flush=True)
                return part
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2 * (attempt+1))

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, range(0, size, chunk)))
    temporary = target.with_suffix(".download")
    with temporary.open("wb") as out:
        for path in paths:
            out.write(path.read_bytes())
    actual = hashlib.md5(temporary.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {name}: {actual}")
    temporary.replace(target)
    print(f"Verified: {name} {size} bytes MD5 {actual}", flush=True)


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        download(name, expected)
