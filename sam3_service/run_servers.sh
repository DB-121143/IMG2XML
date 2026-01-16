#!/usr/bin/env bash
# Start multiple persistent SAM3 HTTP servers (one process per port/GPU).
# You can override env vars below when calling this script.

set -euo pipefail

# ---------- Configurable env vars ----------
# Host binding
HOST=${HOST:-0.0.0.0}
# Device: cuda | cpu
DEVICE=${DEVICE:-cuda}
# Image embedding cache size per server
CACHE_SIZE=${CACHE_SIZE:-0}
# Path to config.yaml
CONFIG=${CONFIG:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/config.yaml"}
# Log directory
LOG_DIR=${LOG_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"}

# ---------- Derived values ----------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${LOG_DIR}"

# ---------- Parse ports from env or config ----------
if [[ -n "${PORTS:-}" ]]; then
  PORT_LIST="${PORTS}"
else
  PORT_LIST=$(CONFIG_PATH="${CONFIG}" python - <<'PY'
import yaml, os, sys
cfg_path = os.environ.get('CONFIG_PATH')
if not cfg_path or not os.path.exists(cfg_path):
  sys.exit("[ERROR] CONFIG_PATH missing or not found")
with open(cfg_path, 'r', encoding='utf-8') as f:
  cfg = yaml.safe_load(f)
ports = []
for ep in cfg.get('services', {}).get('sam3_endpoints', []):
  ep = ep.strip()
  if ep.startswith('http://'):
    ep = ep[len('http://'):]
  if ':' in ep:
    ports.append(ep.split(':', 1)[1])
print(' '.join(ports))
PY
)
fi

if [[ -z "${PORT_LIST}" ]]; then
  echo "[ERROR] No SAM3 ports found. Set PORTS or services.sam3_endpoints in config.yaml" >&2
  exit 1
fi

# ---------- Parse GPUs from env or config (ignored when DEVICE=cpu) ----------
if [[ "${DEVICE}" == "cpu" ]]; then
  GPU_LIST=""
else
  if [[ -n "${GPUS:-}" ]]; then
    GPU_LIST="${GPUS}"
  else
    GPU_LIST=$(CONFIG_PATH="${CONFIG}" python - <<'PY'
import yaml, os, sys
cfg_path = os.environ.get('CONFIG_PATH')
if not cfg_path or not os.path.exists(cfg_path):
    sys.exit("[ERROR] CONFIG_PATH missing or not found")
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
gpus = cfg.get('services', {}).get('sam3_gpus', []) or []
print(' '.join(str(g) for g in gpus))
PY
)
  fi
fi

read -ra PORT_ARR <<<"${PORT_LIST}"
read -ra GPU_ARR <<<"${GPU_LIST:-}"

if [[ "${DEVICE}" != "cpu" ]] && [[ ${#PORT_ARR[@]} -ne ${#GPU_ARR[@]} ]]; then
  echo "[ERROR] DEVICE=cuda 但 PORTS 与 GPUS 数量不一致" >&2
  exit 1
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "CONFIG=${CONFIG}"
echo "HOST=${HOST}"
echo "DEVICE=${DEVICE}"
echo "CACHE_SIZE=${CACHE_SIZE}"
echo "PORTS=${PORT_LIST}"
echo "GPUS=${GPU_LIST:-none}"
echo "Logs -> ${LOG_DIR}"

for idx in "${!PORT_ARR[@]}"; do
  port="${PORT_ARR[$idx]}"
  gpu="${GPU_ARR[$idx]:-}"
  log_file="${LOG_DIR}/server_${port}.log"

  echo "Starting SAM3 server on port ${port} (GPU=${gpu:-none}, device=${DEVICE}) -> ${log_file}"

  if [[ "${DEVICE}" == "cpu" ]]; then
    nohup python -m sam3_service.server \
      --host "${HOST}" \
      --port "${port}" \
      --config "${CONFIG}" \
      --device "cpu" \
      --cache-size "${CACHE_SIZE}" \
      >"${log_file}" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m sam3_service.server \
      --host "${HOST}" \
      --port "${port}" \
      --config "${CONFIG}" \
      --device "cuda" \
      --cache-size "${CACHE_SIZE}" \
      >"${log_file}" 2>&1 &
  fi
  echo "  PID=$!"
done

echo "All servers launched. Use tail -f ${LOG_DIR}/server_<port>.log to view logs."
