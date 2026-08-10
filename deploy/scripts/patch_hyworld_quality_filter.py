#!/usr/bin/env python3
"""Add GSStudio's trajectory whitelist to HY-World's gen_gs_data entrypoint.

The upstream script deliberately discovers every generated trajectory.  For a
quality-gated task that would reintroduce rejected videos, so this narrowly
adds an optional command-line filter without forking the whole upstream file.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys


ARGUMENT_PATTERN = re.compile(
    r"(?m)^(?P<indent>\s*)parser\.add_argument\('--result_name'[^\n]*\)\n"
)
SORT_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)video_paths = sorted\(video_paths, key=lambda x: '\
    r'\(x\.split\("/"\)\[-3\], x\.split\("/"\)\[-2\]\)\)\n'
)
SENTINEL = "# GSStudio trajectory whitelist patch"


def _insert_after(pattern: re.Pattern[str], source: str, insertion_for_indent) -> str:
    match = pattern.search(source)
    if not match:
        raise RuntimeError(f"Expected upstream marker not found: {pattern.pattern}")
    insertion = insertion_for_indent(match.group("indent"))
    return source[:match.end()] + insertion + source[match.end():]


def patch(source: str) -> str:
    if SENTINEL in source:
        return source

    source = _insert_after(
        ARGUMENT_PATTERN,
        source,
        lambda indent: (
            f"{indent}parser.add_argument('--trajectory_whitelist', type=str, default=None, "
            "help='JSON file containing accepted trajectory ids')\n"
        ),
    )
    source = _insert_after(
        SORT_PATTERN,
        source,
        lambda indent: (
            f"{indent}{SENTINEL}\n"
            f"{indent}if args.trajectory_whitelist:\n"
            f"{indent}    with open(args.trajectory_whitelist, 'r') as whitelist_file:\n"
            f"{indent}        whitelist = json.load(whitelist_file)\n"
            f"{indent}    accepted_trajectory_ids = set(whitelist.get('accepted_trajectory_ids', []))\n"
            f"{indent}    video_paths = [\n"
            f"{indent}        video_path for video_path in video_paths\n"
            f"{indent}        if '/'.join(video_path.replace('\\\\', '/').split('/')[-3:-1]) in accepted_trajectory_ids\n"
            f"{indent}    ]\n"
            f"{indent}    rank0_log(f'[Quality] Using {{len(video_paths)}} accepted trajectory videos')\n"
            f"{indent}    if not video_paths:\n"
            f"{indent}        raise RuntimeError('Skipping all videos is not allowed: quality whitelist matched none')\n"
        ),
    )
    return source


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_hyworld_quality_filter.py <gen_gs_data.py>")
    target = Path(sys.argv[1])
    target.write_text(patch(target.read_text()))


if __name__ == "__main__":
    main()
