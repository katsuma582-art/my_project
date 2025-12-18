#!/usr/bin/env python3
"""Bulk rename files in a folder with flexible rules.

Examples:
  Dry-run replace substring:
    python rename_files.py --dir datasets/coins/images --pattern "*.jpg" --replace old new --dry-run

  Add prefix and number files:
    python rename_files.py --dir datasets/coins/images --pattern "*.jpg" --prefix obj_ --number --start 1 --pad 3
"""
import argparse
from pathlib import Path
import fnmatch
import sys


def find_files(directory: Path, pattern: str, recursive: bool):
    if recursive:
        return [p for p in directory.rglob(pattern) if p.is_file()]
    else:
        return [p for p in directory.glob(pattern) if p.is_file()]


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


def main():
    p = argparse.ArgumentParser(description='Bulk rename files (supports dry-run)')
    p.add_argument('--dir', '-d', default='.', help='Target directory')
    p.add_argument('--pattern', default='*', help='Glob pattern (e.g. "*.jpg")')
    p.add_argument('--recursive', action='store_true', help='Search recursively')
    p.add_argument('--dry-run', action='store_true', help='Show actions without renaming')

    group = p.add_mutually_exclusive_group()
    group.add_argument('--replace', nargs=2, metavar=('OLD','NEW'), help='Replace substring in filename')
    group.add_argument('--prefix', help='Add prefix to filename')
    group.add_argument('--suffix', help='Add suffix (before extension)')

    p.add_argument('--number', action='store_true', help='Number matched files sequentially')
    p.add_argument('--start', type=int, default=1, help='Start index for numbering')
    p.add_argument('--pad', type=int, default=0, help='Zero padding for numbering')
    p.add_argument('--lower', action='store_true', help='Convert filename to lowercase')
    p.add_argument('--upper', action='store_true', help='Convert filename to uppercase')
    p.add_argument('--ext', help='Optionally change extension (include leading dot, e.g. .png)')
    p.add_argument('--format', help='Format string for new name. Use {name} for original stem and {num} for sequence number (overrides other name transforms)')
    p.add_argument('--pair', nargs='+', help='Also rename files with these extensions to match new name, e.g. --pair .txt .xml')

    args = p.parse_args()
    directory = Path(args.dir)
    if not directory.exists() or not directory.is_dir():
        print('Directory not found:', directory)
        sys.exit(1)

    files = sorted(find_files(directory, args.pattern, args.recursive))
    if not files:
        print('No files matched.')
        return

    actions = []
    existing_srcs = set()
    idx = args.start
    for f in files:
        name = f.stem
        ext = f.suffix
        new_name = name
        # If user provided a format string, use it. It may include {name} and {num}.
        if args.format:
            if args.number:
                num = str(idx).zfill(args.pad) if args.pad > 0 else str(idx)
                idx += 1
            else:
                num = ''
            try:
                new_name = args.format.format(name=name, num=num)
            except Exception:
                print('Invalid format string. Use {name} and/or {num}.')
                return
            new_ext = args.ext if args.ext is not None else ext
            new_path = f.with_name(new_name + new_ext)
        else:
            if args.replace:
                old, new = args.replace
                new_name = new_name.replace(old, new)
            if args.prefix:
                new_name = args.prefix + new_name
            if args.suffix:
                new_name = new_name + args.suffix
            if args.lower:
                new_name = new_name.lower()
            if args.upper:
                new_name = new_name.upper()
            if args.number:
                num = str(idx).zfill(args.pad) if args.pad > 0 else str(idx)
                new_name = f"{new_name}{num}"
                idx += 1
            new_ext = args.ext if args.ext is not None else ext
            new_path = f.with_name(new_name + new_ext)
        if new_path == f:
            continue
        # ensure no collision
        if new_path.exists():
            new_path = unique_path(new_path)
        # add main rename action if not already added
        if str(f) not in existing_srcs:
            actions.append((f, new_path))
            existing_srcs.add(str(f))
        # handle paired files (e.g. .jpg -> corresponding .txt labels)
        if args.pair:
            for pe in args.pair:
                pe = pe if pe.startswith('.') else f'.{pe}'
                src_pair = f.with_suffix(pe)
                if src_pair.exists() and str(src_pair) not in existing_srcs:
                    dst_pair = new_path.with_suffix(pe)
                    if dst_pair.exists():
                        dst_pair = unique_path(dst_pair)
                    actions.append((src_pair, dst_pair))
                    existing_srcs.add(str(src_pair))

    # show summary
    for src, dst in actions:
        print(f"{src} -> {dst}")

    if not actions:
        print('No changes to perform.')
        return

    if args.dry_run:
        print('\nDry-run: no files were renamed.')
        return

    # perform rename
    for src, dst in actions:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    print(f'Ran {len(actions)} renames.')


if __name__ == '__main__':
    main()