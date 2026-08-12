#!/usr/bin/env bash
# Pre-download all model weights to the persistent model volume.
# Run this as a K8s Job or on a machine with access to the volume.
set -euo pipefail

HF_HOME="${HF_HOME:-/models/hf-cache}"
TORCH_HOME="${TORCH_HOME:-/models/torch-cache}"
export HF_HOME TORCH_HOME
echo "Pre-downloading Hugging Face models to ${HF_HOME}"
echo "Pre-downloading Torch checkpoints to ${TORCH_HOME}"

# Check if running as K8s job or standalone
if [ "${1:-}" = "--k8s" ]; then
    echo "Deploying as K8s Job..."
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: gsstudio-prefetch-models
  namespace: gsstudio
spec:
  backoffLimit: 2
  template:
    spec:
      nodeName: 192.168.80.136
      containers:
        - name: prefetch
          image: ccr-29eug8s3-pub.cnc.bj.baidubce.com/dataflow-poc/gsstudio:a800-v42
          command: ["bash", "-c"]
          args:
            - |
              python -c "
              from huggingface_hub import snapshot_download
              import os
              from pathlib import Path
              from torch.hub import download_url_to_file

              models = [
                  ('facebook/sam3', None),
                  ('naver-iv/zim-anything-vitl', ['zim_vit_l_2092/*']),
                  ('IDEA-Research/grounding-dino-tiny', ['*.json', '*.txt', 'model.safetensors']),
                  ('facebook/dinov2-base', None),
                  # High-quality tasks use the complete WorldStereo checkpoint,
                  # while fast tasks use its DMD variant from the same snapshot.
                  ('hanshanxue/WorldStereo', None),
                  (
                      'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers',
                      [
                          'transformer/*',
                          'text_encoder/*',
                          'image_encoder/*',
                          'image_processor/*',
                          'tokenizer/*',
                          'vae/*',
                          # Full WorldStereo uses UniPC from this directory;
                          # DMD does not, so an incomplete fast-only cache can
                          # otherwise fail only after a long trajectory run.
                          'scheduler/*',
                      ],
                  ),
              ]

              for repo_id, allow in models:
                  print(f'Downloading {repo_id}...')
                  snapshot_download(
                      repo_id,
                      allow_patterns=allow,
                      local_files_only=(repo_id == 'facebook/sam3' and not os.environ.get('HF_TOKEN')),
                  )
                  print(f'Done: {repo_id}')

              checkpoint_dir = Path(os.environ['TORCH_HOME']) / 'hub' / 'checkpoints'
              checkpoint_dir.mkdir(parents=True, exist_ok=True)
              checkpoint = checkpoint_dir / 'vgg16-397923af.pth'
              if not checkpoint.exists():
                  print('Downloading VGG16 perceptual-loss checkpoint...')
                  download_url_to_file(
                      'https://download.pytorch.org/models/vgg16-397923af.pth',
                      str(checkpoint),
                      hash_prefix='397923af',
                      progress=True,
                  )
              print(f'VGG16 checkpoint ready: {checkpoint}')
              print('All models downloaded.')
              "
          env:
            - name: HF_HOME
              value: /models/hf-cache
            - name: TORCH_HOME
              value: /models/torch-cache
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: gsstudio-huggingface
                  key: token
                  optional: true
          volumeMounts:
            - name: models
              mountPath: /models
          resources:
            requests:
              cpu: "4"
              memory: 8Gi
            limits:
              cpu: "8"
              memory: 16Gi
      volumes:
        - name: models
          hostPath:
            path: /home/cce/gsstudio/models
            type: DirectoryOrCreate
      imagePullSecrets:
        - name: gsstudio-registry
      restartPolicy: Never
EOF
    echo "Job created. Monitor with: kubectl -n gsstudio logs -f job/gsstudio-prefetch-models"
    exit 0
fi

# Standalone mode
python3 -c "
from huggingface_hub import snapshot_download
from pathlib import Path
from torch.hub import download_url_to_file
import os

models = [
    'tencent/HY-World-2.0',
    'hanshanxue/WorldStereo',
    'Qwen/Qwen3-VL-8B-Instruct',
    'facebook/sam3',
    'Ruicheng/moge-2-vitl-normal',
    'facebook/dinov2-base',
    'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers',
]

for repo_id in models:
    print(f'Downloading {repo_id}...')
    snapshot_download(repo_id)
    print(f'Done: {repo_id}')

checkpoint_dir = Path(os.environ['TORCH_HOME']) / 'hub' / 'checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint = checkpoint_dir / 'vgg16-397923af.pth'
if not checkpoint.exists():
    print('Downloading VGG16 perceptual-loss checkpoint...')
    download_url_to_file(
        'https://download.pytorch.org/models/vgg16-397923af.pth',
        str(checkpoint),
        hash_prefix='397923af',
        progress=True,
    )
print(f'VGG16 checkpoint ready: {checkpoint}')
print('All models downloaded to', os.environ['HF_HOME'])
"
