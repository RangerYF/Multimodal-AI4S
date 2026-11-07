"""Run existing experiment entry points with immutable provenance snapshots.

Only the Python standard library is needed by this orchestration layer.
Model-specific dependencies remain in the invoked scripts.
"""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = {
    "video": "scripts/run_baseline.py",
    "author_eval": "scripts/evaluate_author_data.py",
    "manual_roi": "scripts/run_manual_roi.py",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def plan(config):
    """Validate a declarative config without loading models or writing files."""
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    run_id = config.get("run_id", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", run_id):
        raise ValueError("run_id must contain only letters, numbers, underscores or hyphens")
    entry = ENTRIES[config["task"]]
    args = config.get("args", [])
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError("args must be a list of strings")
    if any(x == "--output" or x.startswith("--output=") for x in args):
        raise ValueError("output is managed by run_id, not args")
    inputs = config.get("inputs", [])
    if not inputs:
        raise ValueError("Declare input files/directories for fingerprinting")
    for item in inputs:
        if not resolve(item).exists():
            raise FileNotFoundError(item)
    output = ROOT / "outputs" / run_id
    if output.exists():
        raise FileExistsError(f"Run already exists: {output}")
    command = [sys.executable, str(ROOT / entry), *args, "--output", str(output / "artifacts")]
    return output, command


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def fingerprint(inputs):
    records = []
    for value in inputs:
        path = resolve(value)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if file.is_file():
                records.append({"input": value, "file": str(file.relative_to(path)) if path.is_dir() else file.name,
                                "bytes": file.stat().st_size, "sha256": sha256(file)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output, command = plan(config)
    if args.dry_run:
        print(json.dumps({"output": str(output), "command": command}, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    save("config.json", config)
    metadata = {"schema_version": 1, "status": "preparing", "started_at": datetime.now(timezone.utc).isoformat(),
                "command": command, "cwd": str(ROOT), "python": sys.version, "platform": platform.platform(),
                "git_commit": git("rev-parse", "HEAD"),
                "git_status": git("status", "--porcelain", "--untracked-files=all")}
    save("run.json", metadata)
    try:
        # Includes uncommitted source bytes; a dirty worktree is not silently presented as HEAD.
        source_files = []
        for folder in ["src", "scripts"]:
            source_files.extend(sorted((ROOT / folder).rglob("*.py")))
        for file in source_files:
            destination = output / "source" / file.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file.read_bytes())
        save("inputs.json", fingerprint(config["inputs"]))
        save("weights.json", fingerprint(["models/vial.pt", "models/phase.pt"]))
        metadata["external_versions"] = {
            name: git("-C", str(ROOT / "external" / name), "rev-parse", "HEAD")
            for name in ["yolov5", "CV-HTE-Tutorial"]}
        with (output / "environment.txt").open("w", encoding="utf-8") as env:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=env, stderr=subprocess.STDOUT, check=True)
        metadata["status"] = "running"
        save("run.json", metadata)
        with (output / "console.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        metadata.update(returncode=result.returncode, status="completed" if result.returncode == 0 else "failed")
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        save("run.json", metadata)
    print(f"{metadata['status']}: {output}")
    return metadata["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
