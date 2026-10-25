#!/usr/bin/env python3
# Copyright 2026 GSStudio contributors.
# Original patch-tool code: Apache-2.0. Upstream-derived matching excerpts
# retain the Tencent HY-WORLD 2.0 Community License; see THIRD_PARTY_NOTICES.md.
# Source: Tencent-Hunyuan/HY-World-2.0, hyworld2/worldgen/gen_gs_data.py
# Original license: https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/License.txt
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
MODIFICATION_NOTICE = (
    "# GSStudio modification notice (2026-10-25): modified by GSStudio contributors.\n"
    "# Changes: trajectory whitelist and panorama/polar anchor fallback.\n"
    "# Original source: Tencent-Hunyuan/HY-World-2.0, gen_gs_data.py.\n"
    "# Original license: https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/License.txt\n"
)


def _with_modification_notice(source: str) -> str:
    if MODIFICATION_NOTICE in source:
        return source
    lines = source.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines[:2]):
        if (index == 0 and line.startswith("#!")) or re.match(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+", line):
            offset = sum(len(item) for item in lines[:index + 1])
    return source[:offset] + MODIFICATION_NOTICE + source[offset:]


def _insert_after(pattern: re.Pattern[str], source: str, insertion_for_indent) -> str:
    match = pattern.search(source)
    if not match:
        raise RuntimeError(f"Expected upstream marker not found: {pattern.pattern}")
    insertion = insertion_for_indent(match.group("indent"))
    return source[:match.end()] + insertion + source[match.end():]


def patch(source: str) -> str:
    source = _with_modification_notice(source)
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
            f"{indent}        rank0_log('[Quality] Empty trajectory whitelist: using panorama/polar RGB-D anchors only')\n"
            f"{indent}        anchor_resolution_sources = (\n"
            f"{indent}            sorted(glob(f'{{pano_bank_path}}/images/*.png')) +\n"
            f"{indent}            sorted(glob(f'{{polar_bank_path}}/images/*.png'))\n"
            f"{indent}        )\n"
            f"{indent}        if not anchor_resolution_sources:\n"
            f"{indent}            raise RuntimeError('Anchored 3DGS requires panorama or polar RGB-D source views')\n"
            f"{indent}        with Image.open(anchor_resolution_sources[0]) as anchor_image:\n"
            f"{indent}            img_width, img_height = anchor_image.size\n"
            f"{indent}        rank0_log(f'[Quality] Anchor output resolution: {{img_width}}x{{img_height}}')\n"
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
