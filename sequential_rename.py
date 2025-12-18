#!/usr/bin/env python3
"""Rename files in a folder to sequential names.

Examples:
  Dry-run, zero-padded 3 digits, start at 0:
    python sequential_rename.py --dir datasets/all_data/images --pattern "*.jpg" --prefix sample --start 0 --pad 3 --dry-run

  Apply changes:
    python sequential_rename.py --dir datasets/all_data/images --pattern "*.jpg" --prefix sample --start 0 --pad 3
"""
from pathlib import Path
import argparse
import sys


def unique_path(p: Path):
    if not p.exists():
        return p
    stem = p.stem
    suffix = p.suffix
    i = 1
    while True:
        np = p.with_name(f"{stem}_{i}{suffix}")
        if not np.exists():
            return np
        i += 1


def find_files(directory: Path, pattern: str, recursive: bool):
    if recursive:
        return sorted([p for p in directory.rglob(pattern) if p.is_file()])
    return sorted([p for p in directory.glob(pattern) if p.is_file()])


def main():
    p = argparse.ArgumentParser(description='Sequentially rename files in a directory')
    p.add_argument('--dir', '-d', default='.', help='Target directory')
    p.add_argument('--pattern', default='*', help='Glob pattern to match files (e.g. "*.jpg")')
    p.add_argument('--recursive', action='store_true', help='Search recursively')
    p.add_argument('--prefix', default='file', help='Prefix for new filenames')
    p.add_argument('--start', type=int, default=0, help='Start index (default 0)')
    p.add_argument('--pad', type=int, default=0, help='Zero pad width for numbers')
    p.add_argument('--ext', help='Optional extension for new files (include leading dot), otherwise keep original')
    p.add_argument('--dry-run', action='store_true', help='Show planned renames without applying')

    args = p.parse_args()
    directory = Path(args.dir)
    if not directory.exists() or not directory.is_dir():
        print('Directory not found:', directory)
        sys.exit(1)

    files = find_files(directory, args.pattern, args.recursive)
    if not files:
        print('No files matched.')
        return

    actions = []
    idx = args.start
    for f in files:
        new_num = str(idx).zfill(args.pad) if args.pad > 0 else str(idx)
        new_stem = f"{args.prefix}{new_num}"
        new_ext = args.ext if args.ext is not None else f.suffix
        dst = f.with_name(new_stem + new_ext)
        if dst.exists():
            dst = unique_path(dst)
        if dst == f:
            idx += 1
            continue
        actions.append((f, dst))
        idx += 1

    for s, d in actions:
        print(f"{s} -> {d}")

    if not actions:
        print('No changes to perform.')
        return

    if args.dry_run:
        print('\nDry-run: no files were renamed.')
        return

    for s, d in actions:
        d.parent.mkdir(parents=True, exist_ok=True)
        s.rename(d)
    print(f'Ran {len(actions)} renames.')


if __name__ == '__main__':
    main()
