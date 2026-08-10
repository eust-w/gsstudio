#!/usr/bin/env python3
"""Add a deterministic candidate whitelist to HY-World's video generator.

WorldStereo's FSDP ranks cooperate on one trajectory at a time.  Applying the
quality-selected candidate set before that loop bounds high-quality latency
without weakening the later alignment whitelist used by ``gen_gs_data``.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys


ARGUMENT_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)parser\.add_argument\("--target_path"[^\n]*\)\n'
)
SORT_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)render_list = sort_trajs\(f"\{scene\}/render_results"\)\n'
)
SENTINEL = "# GSStudio video candidate whitelist patch"


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
            f'{indent}parser.add_argument("--trajectory_whitelist", type=str, default=None, '
            'help="JSON file containing planned high-quality trajectory ids")\n'
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
            f"{indent}    candidate_trajectory_ids = set(whitelist.get('candidate_trajectory_ids', []))\n"
            f"{indent}    render_list = [\n"
            f"{indent}        render_path for render_path in render_list\n"
            f"{indent}        if '/'.join(render_path.replace('\\\\', '/').split('/')[-3:-1]) in candidate_trajectory_ids\n"
            f"{indent}    ]\n"
            f"{indent}    rank0_log(f'[Quality] Using {{len(render_list)}} planned high-quality trajectories')\n"
            f"{indent}    if not render_list:\n"
            f"{indent}        raise RuntimeError('No trajectory renderings matched the high-quality candidate whitelist')\n"
        ),
    )
    return source


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_hyworld_video_filter.py <video_gen.py>")
    target = Path(sys.argv[1])
    target.write_text(patch(target.read_text()))


if __name__ == "__main__":
    main()
