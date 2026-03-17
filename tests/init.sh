#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ---- Config (fill in before running) ----
namespace=""
hf_token=""
storage_class_name="default"
port_local=8080
port_remote=8000
MODEL_NAME="facebook/opt-125m"
max_model_len=1000
lmcache_enabled=true
lmcache_chunk_size=256
lmcache_local_cpu=true
lmcache_max_local_cpu_size=20

[ -z "$namespace" ] && { echo "Error: set namespace in init.sh" >&2; exit 1; }
[ -z "$hf_token" ] && { echo "Error: set hf_token in init.sh" >&2; exit 1; }

if [ "${1:-}" = "--cleanup" ]; then
    pkill -f "kubectl.*port-forward" 2>/dev/null || true
    kubectl -n "$namespace" delete job vllm --ignore-not-found
    kubectl -n "$namespace" delete pod -l app=vllm --ignore-not-found --force --grace-period=0 2>/dev/null || true
    kubectl -n "$namespace" delete svc -l app=vllm --ignore-not-found
    kubectl -n "$namespace" delete configmap vllm-config --ignore-not-found
    kubectl -n "$namespace" delete secret hf-token-secret --ignore-not-found
    [ -d "k8s" ] && rm -rf k8s
    echo "Done."
    exit 0
fi

echo "Launching $MODEL_NAME in $namespace (localhost:${port_local} -> pod:${port_remote})"

lmcache_literals=""
if [ "$lmcache_enabled" = "true" ]; then
    lmcache_literals="
      - LMCACHE_ENABLED=true
      - LMCACHE_CHUNK_SIZE=${lmcache_chunk_size}
      - LMCACHE_LOCAL_CPU=${lmcache_local_cpu}
      - LMCACHE_MAX_LOCAL_CPU_SIZE=${lmcache_max_local_cpu_size}"
else
    lmcache_literals="
      - LMCACHE_ENABLED=false"
fi

mkdir -p k8s
cp template/* k8s

cat > k8s/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: ${namespace}

resources:
  - job.yaml
  - pvc.yaml
  - svc.yaml

configMapGenerator:
  - name: vllm-config
    literals:
      - MODEL=${MODEL_NAME}
      - MAX_MODEL_LEN=${max_model_len}
      - VLLM_USE_FLASHINFER_SAMPLER=0
      - VLLM_LOGGING_LEVEL=DEBUG
      - VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=1
      - VLLM_ATTENTION_BACKEND=TORCH_SDPA${lmcache_literals}
    options:
      disableNameSuffixHash: true

secretGenerator:
  - name: hf-token-secret
    literals:
      - token=${hf_token}
    options:
      disableNameSuffixHash: true

replacements:
  - source:
      kind: Secret
      name: hf-token-secret
      fieldPath: metadata.name
    targets:
      - select: { kind: Job, name: vllm }
        fieldPaths:
          - spec.template.spec.containers.[name=vllm-container].env.[name=HUGGING_FACE_HUB_TOKEN].valueFrom.secretKeyRef.name
          - spec.template.spec.containers.[name=vllm-container].env.[name=HF_TOKEN].valueFrom.secretKeyRef.name
          - spec.template.spec.containers.[name=infinity].env.[name=HUGGING_FACE_HUB_TOKEN].valueFrom.secretKeyRef.name
          - spec.template.spec.containers.[name=infinity].env.[name=HF_TOKEN].valueFrom.secretKeyRef.name

patchesJson6902:
  - target:
      group: v1
      version: v1
      kind: PersistentVolumeClaim
      name: vllm-pvc
    patch: |-
      - op: replace
        path: /spec/storageClassName
        value: ${storage_class_name}
  - target:
      group: v1
      version: v1
      kind: PersistentVolumeClaim
      name: hfcache-pvc
    patch: |-
      - op: replace
        path: /spec/storageClassName
        value: ${storage_class_name}
EOF

kubectl kustomize k8s > k8s/manifests.yaml

kubectl -n "$namespace" delete job vllm --ignore-not-found
kubectl -n "$namespace" delete pod -l app=vllm --ignore-not-found --force --grace-period=0 2>/dev/null || true
kubectl -n "$namespace" wait --for=delete job/vllm --timeout=120s 2>/dev/null || true
kubectl apply -f k8s/manifests.yaml || true

echo "Waiting for pod to be ready..."
kubectl -n "$namespace" wait --for=condition=Ready pod -l app=vllm --timeout=30m

POD="$(kubectl -n "$namespace" get pod -l app=vllm -o jsonpath='{.items[0].metadata.name}')"
pkill -f "port-forward.*${port_local}:${port_remote}" 2>/dev/null || true
sleep 2

echo ""
echo "  Endpoint: http://127.0.0.1:${port_local}"
echo "  Model:    ${MODEL_NAME}"
echo "  Run:      python main.py narrativeqa humaneval"
echo ""

kubectl -n "$namespace" port-forward "pod/${POD}" "${port_local}:${port_remote}"

