#!/usr/bin/env python3
# Copyright 2026 GSStudio contributors.
# Original patch-tool code: Apache-2.0. Upstream-derived matching excerpts
# retain the Tencent HY-WORLD 2.0 Community License; see THIRD_PARTY_NOTICES.md.
# Source: Tencent-Hunyuan/HY-World-2.0, hyworld2/worldgen/models/worldstereo_wrapper.py
# Original license: https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/License.txt
"""Make HY-World's WorldStereo transformer honor local_files_only."""
from __future__ import annotations

import re
import sys
from pathlib import Path


MODIFICATION_NOTICE = (
    "# GSStudio modification notice (2026-10-25): modified by GSStudio contributors.\n"
    "# Changes: local-only model loading and T5 tokenizer compatibility.\n"
    "# Original source: Tencent-Hunyuan/HY-World-2.0, worldstereo_wrapper.py.\n"
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


def replace_once(source: str, old: str, new: str) -> str:
    old_count = source.count(old)
    new_count = source.count(new)
    if old_count == 1:
        return source.replace(old, new, 1)
    if old_count == 0 and new_count == 1:
        return source
    raise RuntimeError(
        f"Expected one upstream or patched match, found old={old_count} new={new_count}: {old!r}"
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} WORLDSTEREO_WRAPPER_PATH")

    path = Path(sys.argv[1])
    source = path.read_text()

    source = replace_once(
        source,
        "from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel",
        "from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, T5Tokenizer, UMT5EncoderModel",
    )

    source = replace_once(
        source,
        """            device_mesh=device_mesh,
            device=device,
        )

        text_encoder, image_clip, vae = cls._load_aux(""",
        """            device_mesh=device_mesh,
            device=device,
            local_files_only=local_files_only,
        )

        text_encoder, image_clip, vae = cls._load_aux(""",
    )
    source = replace_once(
        source,
        """        device_mesh,
        device,
    ):

        half_dtype = _get_half_dtype()""",
        """        device_mesh,
        device,
        local_files_only: bool,
    ):

        half_dtype = _get_half_dtype()""",
    )
    source = replace_once(
        source,
        """                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
            )
        else:""",
        """                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
                local_files_only=local_files_only,
            )
        else:""",
    )
    source = replace_once(
        source,
        """                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
            )

        rank0_log(\"Building ControlNet…\")""",
        """                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
                local_files_only=local_files_only,
            )

        rank0_log(\"Building ControlNet…\")""",
    )
    source = replace_once(
        source,
        'tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)',
        'tokenizer = T5Tokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)',
    )

    path.write_text(_with_modification_notice(source))


if __name__ == "__main__":
    main()
