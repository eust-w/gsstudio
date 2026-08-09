#!/usr/bin/env bash
# Pre-download all model weights to the HF cache PVC.
# Run this as a K8s Job or on a machine with access to the PVC.
set -euo pipefail

HF_HOME="${HF_HOME:-/models/hf-cache}"
echo "Pre-downloading models to ${HF_HOME}"

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
          image: ccr-29eug8s3-pub.cnc.bj.baidubce.com/dataflow-poc/gsstudio:a800-v16
          command: ["bash", "-c"]
          args:
            - |
              python -c "
              from huggingface_hub import snapshot_download
              import os

              models = [
                  ('facebook/sam3', None),
                  ('naver-iv/zim-anything-vitl', ['zim_vit_l_2092/*']),
                  ('IDEA-Research/grounding-dino-tiny', ['*.json', '*.txt', 'model.safetensors']),
                  ('facebook/dinov2-base', None),
                  (
                      'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers',
                      [
                          'transformer/*',
                          'text_encoder/*',
                          'image_encoder/*',
                          'image_processor/*',
                          'tokenizer/*',
                          'vae/*',
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

              print('All models downloaded.')
              "
          env:
            - name: HF_HOME
              value: /models/hf-cache
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: gsstudio-huggingface
                  key: token
                  optional: true
          volumeMounts:
            - name: hf-cache
              mountPath: /models/hf-cache
          resources:
            requests:
              cpu: "4"
              memory: 8Gi
            limits:
              cpu: "8"
              memory: 16Gi
      volumes:
        - name: hf-cache
          hostPath:
            path: /home/cce/gsstudio/models/hf-cache
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

print('All models downloaded to', '${HF_HOME}')
"
