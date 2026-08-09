#!/usr/bin/env python3
"""Make HY-World's WorldStereo transformer honor local_files_only."""
from __future__ import annotations

import sys
from pathlib import Path


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

    path.write_text(source)


if __name__ == "__main__":
    main()
