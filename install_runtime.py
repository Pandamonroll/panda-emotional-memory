from __future__ import annotations

import os
from pathlib import Path
import re
from zipfile import ZipFile
import shutil
import tempfile
import json
import urllib.parse
import urllib.request
import uuid

from pip._internal.cli.main import main as pip_main


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv-real"
SITE_PACKAGES = VENV_DIR / "Lib" / "site-packages"
WHEELHOUSE = ROOT / ".wheelhouse"
FIXEDTMP = ROOT / ".fixedtmp"
MODELS_DIR = ROOT / ".models"
DEFAULT_MODEL_REPO = os.environ.get(
    "MEMORY_MODEL_REPO",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

CORE_REQUIREMENTS = [
    "torch==2.7.0",
    "transformers",
]
OPTIONAL_REQUIREMENTS = {
    "sentencepiece": ["sentencepiece"],
}


def patch_tempfile() -> None:
    FIXEDTMP.mkdir(parents=True, exist_ok=True)

    def custom_mkdtemp(*args, **kwargs) -> str:
        path = FIXEDTMP / f"tmp-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        return str(path)

    class CustomTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self.name = custom_mkdtemp()

        def __enter__(self) -> str:
            return self.name

        def __exit__(self, exc_type, exc, tb) -> None:
            shutil.rmtree(self.name, ignore_errors=True)

        def cleanup(self) -> None:
            shutil.rmtree(self.name, ignore_errors=True)

    tempfile.mkdtemp = custom_mkdtemp
    tempfile.TemporaryDirectory = CustomTemporaryDirectory


def ensure_system_site_packages() -> None:
    cfg = VENV_DIR / "pyvenv.cfg"
    text = cfg.read_text(encoding="utf-8")
    if "include-system-site-packages = false" in text:
        text = text.replace(
            "include-system-site-packages = false",
            "include-system-site-packages = true",
        )
        cfg.write_text(text, encoding="utf-8")


def requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].replace("-", "_").lower()


def has_cached_wheel(requirement: str) -> bool:
    normalized = requirement_name(requirement)
    candidates = {f"{normalized}-", f"{normalized.replace('_', '-')}-"}
    for wheel in WHEELHOUSE.glob("*.whl"):
        lower_name = wheel.name.lower()
        if any(lower_name.startswith(candidate) for candidate in candidates):
            return True
    return False


def download_wheels(requirements: list[str]) -> None:
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    missing = [requirement for requirement in requirements if not has_cached_wheel(requirement)]
    if not missing:
        return
    args = [
        "download",
        "--dest",
        str(WHEELHOUSE),
        "--no-cache-dir",
        "--disable-pip-version-check",
        *missing,
    ]
    result = pip_main(args)
    if result != 0:
        raise SystemExit(result)


def wheel_member_target(member: str) -> tuple[str, Path] | None:
    parts = Path(member).parts
    if ".data" not in parts:
        return "site", Path(*parts)

    idx = parts.index(".data")
    if idx + 1 >= len(parts):
        return None

    section = parts[idx + 1]
    remainder = Path(*parts[idx + 2 :])
    if section in {"purelib", "platlib"}:
        return "site", remainder
    if section == "scripts":
        return "scripts", remainder
    return None


def install_wheel(wheel_path: Path) -> None:
    scripts_dir = VENV_DIR / "Scripts"
    with ZipFile(wheel_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            mapped = wheel_member_target(info.filename)
            if mapped is None:
                continue

            location, relative_path = mapped
            if location == "site":
                destination = SITE_PACKAGES / relative_path
            else:
                destination = scripts_dir / relative_path

            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)


def install_all_downloaded_wheels() -> None:
    for wheel_path in sorted(WHEELHOUSE.glob("*.whl")):
        install_wheel(wheel_path)


def download_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request) as response, open(destination, "wb") as output:
        shutil.copyfileobj(response, output)


def model_snapshot_complete(target_dir: Path) -> bool:
    if not target_dir.exists():
        return False
    required_files = [
        target_dir / "config.json",
        target_dir / "special_tokens_map.json",
    ]
    weight_files = [
        target_dir / "model.safetensors",
        target_dir / "pytorch_model.bin",
        target_dir / "onnx" / "model.onnx",
    ]
    return all(path.exists() for path in required_files) and any(path.exists() for path in weight_files)


def download_model_snapshot(repo_id: str) -> Path:
    local_name = repo_id.replace("/", "--")
    target_dir = MODELS_DIR / local_name
    target_dir.mkdir(parents=True, exist_ok=True)
    if model_snapshot_complete(target_dir):
        return target_dir

    api_url = f"https://huggingface.co/api/models/{repo_id}"
    data = download_json(api_url)

    for sibling in data.get("siblings", []):
        relative_name = sibling["rfilename"]
        if relative_name == ".gitattributes":
            continue

        destination = target_dir / relative_name
        if destination.exists():
            continue

        quoted_name = urllib.parse.quote(relative_name, safe="/")
        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{quoted_name}?download=1"
        download_file(file_url, destination)

    return target_dir


def selected_requirements() -> list[str]:
    requirements = list(CORE_REQUIREMENTS)
    if os.environ.get("MEMORY_INSTALL_SENTENCEPIECE") == "1":
        requirements.extend(OPTIONAL_REQUIREMENTS["sentencepiece"])
    return requirements


def main() -> None:
    patch_tempfile()
    ensure_system_site_packages()
    download_wheels(selected_requirements())
    install_all_downloaded_wheels()
    model_dir = download_model_snapshot(DEFAULT_MODEL_REPO)
    print("Installed runtime packages into .venv-real")
    print(f"Downloaded model snapshot to {model_dir}")
    if os.environ.get("MEMORY_INSTALL_SENTENCEPIECE") != "1":
        print("Skipped optional sentencepiece helper. Set MEMORY_INSTALL_SENTENCEPIECE=1 to fetch it.")


if __name__ == "__main__":
    main()
