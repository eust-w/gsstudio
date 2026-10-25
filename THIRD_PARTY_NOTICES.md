# Third-party notices

Copyright 2026 GSStudio contributors.

The [Apache-2.0 license](LICENSE) covers original GSStudio application code,
documentation, and original patch-tool logic. It does **not** relicense third-party
code, upstream-derived excerpts, model weights, or runtime components. Source
matching excerpts in `deploy/scripts/patch_hyworld_*.py` and any upstream-derived
test excerpts retain the applicable upstream terms. Applying a patch does not
replace the license of the modified upstream file.

This source release does not bundle model weights, a complete HY-World checkout,
or a GPU runtime. Mock mode runs without those components. Real inference requires
the operator to obtain the upstream components separately and comply with their
licenses. The independent GSStudio application and its optional model integrations
have different licensing scopes.

## HY-World 2.0 and HY-Pano 2.0

- Upstream: [Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0).
- License: [Tencent HY-WORLD 2.0 Community License Agreement, including its AUP](third_party/licenses/HY-World-2.0.txt).
- Model source: [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0),
  including its `HY-Pano-2.0` directory.
- Integration: external panorama, trajectory, WorldStereo, WorldMirror, and 3DGS
  commands. The deployment patches modify local loading, trajectory selection,
  resume behavior, and render reuse in a separately obtained upstream checkout.

This is a custom community license, not Apache-2.0. Its territory excludes the
European Union, United Kingdom, and South Korea. Section 4 uses a **1 million
monthly active user** threshold, measured for the preceding calendar month at the
upstream version's release date, for additional commercial authorization. The
license also contains use restrictions, including restrictions involving other
AI models and outputs. Read the complete agreement for the applicable scope.

Redistributors of covered material must carry the upstream agreement and its
required notice, retain applicable notices, and mark modified files. The exact
notice from section 3(d) is included in [NOTICE](NOTICE). The GSStudio patch scripts
add a prominent modification notice while preserving existing upstream content.
Service providers must disclose their actual full legal name/entity and state
that Tencent is not affiliated with, associated with, sponsoring, or endorsing
their service, as required by the upstream agreement.

## HunyuanImage-3.0

- Upstream: [Tencent-Hunyuan/HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0).
- License: [Tencent Hunyuan Community License Agreement, including its AUP](third_party/licenses/HunyuanImage-3.0.txt).
- Integration: the HunyuanImage-3 backend used by HY-Pano, including prompt-source
  generation in `server/pipeline/panogen_adapter.py`.
- Attribution evidence: the [upstream model implementation's file header](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/hyworld2/panogen/hunyuan_image_3/modeling_hunyuan_image_3.py)
  explicitly references the HunyuanImage-3.0 agreement.

This separate community license also has territory and use restrictions. Its
additional commercial-authorization threshold is **100 million monthly active
users**, not HY-World's 1 million threshold. Its agreement, AUP, attribution,
modification, provider-identification, and distribution requirements remain
applicable to covered components. Its required section 3(d) text is also included
in [NOTICE](NOTICE). The summary here does not replace either agreement or promise
authorization for a particular deployment.

## Other optional model/runtime components

The existing GPU integration references the following components. They are not
included in this source release. These references are not a complete software
bill of materials for an independently assembled GPU image.

| Component | Upstream license/source | Role |
| --- | --- | --- |
| SAM3 | [SAM License](third_party/licenses/SAM3.txt); [official repository](https://github.com/facebookresearch/sam3) | Segmentation; custom license with separate use and redistribution terms |
| Qwen3-VL-8B-Instruct | [Official model card: Apache-2.0](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | VLM used for planning/captioning |
| MoGe-2 normal model | [Model card: MIT](https://huggingface.co/Ruicheng/moge-2-vitl-normal); [MoGe source license and included dependency terms](https://github.com/microsoft/MoGe/blob/main/LICENSE) | Geometry/normal estimation |
| DINOv2 | [Official Apache-2.0 license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | Vision backbone |
| Wan2.1 I2V 14B 480P | [Official model card: Apache-2.0](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) | WorldStereo base-model components |
| Legacy `hanshanxue/WorldStereo` checkpoint source | [Model card metadata: MIT](https://huggingface.co/hanshanxue/WorldStereo) | Existing checkpoint-download compatibility; the card has no descriptive README, so its metadata is not treated as permission to relicense the HY-World runtime |

Additional components installed through upstream requirements, such as CUDA,
PyTorch, native extensions, VGG16 checkpoints, and segmentation dependencies,
retain their own terms. A distributor who assembles a binary image must inventory
the actual versions and preserve the notices required by those components.

## Browser and application dependencies

The browser viewer uses [Spark](https://github.com/sparkjsdev/spark), copyright
WORLD LABS TECHNOLOGIES, INC., under the [MIT license](third_party/licenses/Spark.txt).
React, React DOM, React Router, Three.js, and three-mesh-bvh are separate packages;
their package versions are recorded in `web/package-lock.json` and their own
license/copyright files remain applicable. Python dependencies likewise retain
their package licenses. The Apache license for GSStudio does not replace any
dependency's terms. Distributors of compiled web assets or container images must
retain the applicable dependency license and copyright notices.

## Provenance and project identity

The verbatim license copies were retrieved on **2026-10-25**. Their exact URLs,
upstream commits, and SHA-256 hashes are recorded in
[third_party/licenses/sources.json](third_party/licenses/sources.json), with a
[human-readable index](third_party/licenses/README.md). Those commits identify
the reviewed license sources; they are not claims that the GPU pipeline was
validated against those revisions. Consult the upstream terms for the exact
component version that you obtain.

GSStudio is an independent project. Tencent is not affiliated with, associated
with, sponsoring, or endorsing GSStudio. No Tencent or other third-party trademark
rights are granted by GSStudio's Apache-2.0 license.
