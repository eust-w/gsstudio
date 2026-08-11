#!/usr/bin/env python3
"""Patch HY-World trajectory rendering to accept a recovery whitelist.

The upstream renderer renders and captions every planned trajectory.  Recovery
adds only six safe paths, so repeating the entire planner output would waste
GPU time and vLLM requests.  The patch is marker-based and idempotent because
the release image can be rebuilt from already-patched parents.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys


ARGUMENT_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)parser\.add_argument\("--target_path"[^\n]*\)\n'
)
ARGS_PARSE_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)args = parser\.parse_args\(\)\n'
)
RANK_LOG_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)rank0_log\(f"Get \{len\(traj_list\)\} trajectories at all for \{scene_path\}"\)\n'
)
CAPTION_LIST_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)total_render_list = glob\(f"\{scene_path\}/render_results/\*/traj\*/render\.mp4"\)\n'
)
CAMERA_GUARD_PATTERN = re.compile(
    r'(?m)^(?P<indent>\s*)if not os\.path\.exists\(f"\{traj_path\}/camera\.json"\):\n'
    r'(?P=indent)    continue\n'
)
ARGUMENT_SENTINEL = "# GSStudio trajectory render whitelist patch"
LOAD_SENTINEL = "# GSStudio trajectory whitelist load patch"
TRAJECTORY_SENTINEL = "# GSStudio trajectory list whitelist patch"
CAPTION_SENTINEL = "# GSStudio trajectory caption whitelist patch"
SKIP_SENTINEL = "# GSStudio trajectory render resume patch"


def _insert_after(pattern: re.Pattern[str], source: str, insertion_for_indent) -> str:
    match = pattern.search(source)
    if not match:
        raise RuntimeError(f"Expected upstream marker not found: {pattern.pattern}")
    insertion = insertion_for_indent(match.group("indent"))
    return source[:match.end()] + insertion + source[match.end():]


def _insert_before(pattern: re.Pattern[str], source: str, insertion_for_indent) -> str:
    match = pattern.search(source)
    if not match:
        raise RuntimeError(f"Expected upstream marker not found: {pattern.pattern}")
    insertion = insertion_for_indent(match.group("indent"))
    return source[:match.start()] + insertion + source[match.start():]


def patch(source: str) -> str:
    if ARGUMENT_SENTINEL not in source:
        source = _insert_after(
            ARGUMENT_PATTERN,
            source,
            lambda indent: (
                f"{indent}{ARGUMENT_SENTINEL}\n"
                f'{indent}parser.add_argument("--trajectory_whitelist", type=str, default=None, '
                'help="JSON file containing only trajectories to render")\n'
                f'{indent}parser.add_argument("--skip_exist", action="store_true", '
                'help="Reuse rendered recovery trajectories when present")\n'
            ),
        )

    # Patch sections are intentionally independent. Release images can inherit
    # a legacy patch that defined candidate_trajectory_ids *after* captioning.
    # Load it immediately after argument parsing so every rank has the value
    # before either rendering or rank-zero caption filtering.
    if LOAD_SENTINEL not in source:
        source = _insert_after(
            ARGS_PARSE_PATTERN,
            source,
            lambda indent: (
                f"{indent}{LOAD_SENTINEL}\n"
                f"{indent}candidate_trajectory_ids = set()\n"
                f"{indent}if args.trajectory_whitelist:\n"
                f"{indent}    with open(args.trajectory_whitelist, 'r') as whitelist_file:\n"
                f"{indent}        whitelist = json.load(whitelist_file)\n"
                f"{indent}    candidate_trajectory_ids = set(whitelist.get('candidate_trajectory_ids', []))\n"
            ),
        )

    # Filter immediately before the upstream count/log marker. This anchor is
    # stable across the four-line and future arbitrary trajectory glob lists,
    # unlike the previous parenthesis-based expression matcher.
    if TRAJECTORY_SENTINEL not in source:
        source = _insert_before(
            RANK_LOG_PATTERN,
            source,
            lambda indent: (
                f"{indent}{TRAJECTORY_SENTINEL}\n"
                f"{indent}if args.trajectory_whitelist:\n"
                f"{indent}    traj_list = [\n"
                f"{indent}        traj_path for traj_path in traj_list\n"
                f"{indent}        if '/'.join(traj_path.replace('\\\\', '/').rstrip('/').split('/')[-2:]) in candidate_trajectory_ids\n"
                f"{indent}    ]\n"
                f"{indent}    if not traj_list:\n"
                f"{indent}        raise RuntimeError('No trajectory paths matched the recovery whitelist')\n"
            ),
        )

    if SKIP_SENTINEL not in source:
        guard = CAMERA_GUARD_PATTERN.search(source)
        if not guard:
            raise RuntimeError("Expected upstream camera guard not found")
        insertion = (
            f"{guard.group('indent')}if args.skip_exist and os.path.exists(f\"{{traj_path}}/render.mp4\") "
            f"and os.path.exists(f\"{{traj_path}}/render_mask.mp4\"):  {SKIP_SENTINEL}\n"
            f"{guard.group('indent')}    continue\n"
        )
        source = source[:guard.end()] + insertion + source[guard.end():]

    if CAPTION_SENTINEL not in source:
        source = _insert_after(
            CAPTION_LIST_PATTERN,
            source,
            lambda indent: (
                f"{indent}{CAPTION_SENTINEL}\n"
                f"{indent}if args.trajectory_whitelist:\n"
                f"{indent}    total_render_list = [\n"
                f"{indent}        render_path for render_path in total_render_list\n"
                f"{indent}        if '/'.join(render_path.replace('\\\\', '/').split('/')[-3:-1]) in candidate_trajectory_ids\n"
                f"{indent}    ]\n"
            ),
        )
    return source


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_hyworld_render_filter.py <traj_render.py>")
    target = Path(sys.argv[1])
    target.write_text(patch(target.read_text()))


if __name__ == "__main__":
    main()
