from __future__ import annotations

import sys

from install_runtime import DEFAULT_MODEL_REPO, download_model_snapshot


def main() -> None:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_REPO
    target = download_model_snapshot(repo_id)
    print(target)


if __name__ == "__main__":
    main()
