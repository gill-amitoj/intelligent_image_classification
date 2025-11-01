# ----------------------------------------------------------
# Validate and clean images in an ImageFolder directory
# ----------------------------------------------------------
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def iter_image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def is_image_ok(path: Path) -> Tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()  # verify header
        # Reopen to catch some lazy errors
        with Image.open(path) as img:
            img.load()
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Validate images under an ImageFolder root.")
    parser.add_argument("--root", type=str, required=True, help="Root folder (e.g., data/train or data/val)")
    parser.add_argument(
        "--action",
        type=str,
        default="move",
        choices=["report", "move", "remove"],
        help="What to do with corrupted images (default: move)",
    )
    parser.add_argument(
        "--corrupted_dir",
        type=str,
        default=None,
        help="Directory to move corrupted images to (defaults to <root>/_corrupted)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}")
        return

    corrupted_dir = Path(args.corrupted_dir) if args.corrupted_dir else (root / "_corrupted")
    if args.action == "move":
        corrupted_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    bad = 0
    for img_path in iter_image_files(root):
        total += 1
        ok, err = is_image_ok(img_path)
        if not ok:
            bad += 1
            rel = img_path.relative_to(root)
            print(f"[BAD] {rel} -> {err}")
            if args.action == "remove":
                img_path.unlink(missing_ok=True)
            elif args.action == "move":
                dest = corrupted_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dest))
            # report: do nothing

    print(f"Checked: {total} files | Corrupted: {bad}")


if __name__ == "__main__":
    main()
