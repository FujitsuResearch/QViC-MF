#!/usr/bin/env bash
set -euo pipefail

export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

if ! python -c "import lmms_eval" >/dev/null 2>&1; then
    echo "[ERROR] 'lmms_eval' is not importable from $(command -v python)." >&2
    echo "        Activate the venv first:  source .venv/bin/activate" >&2
    echo "        Or install dependencies:  pip install -e .[eval]" >&2
    exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] 'accelerate' is not on PATH. Activate the venv first:" >&2
    echo "        source .venv/bin/activate" >&2
    exit 1
fi

################

pretrained="ckpt/QViC-MF-7B"  # TODO: switch back to "Fujitsu/QViC-MF-7B" once published on HF

model_base="lmms-lab/LLaVA-Video-7B-Qwen2"
conv_template="qwen_1_5"
context_memory_length=256
min_frames_num=64
max_frames_num=-1
fps=2

parameter="pretrained=${pretrained},model_base=${model_base},conv_template=${conv_template},context_memory_length=${context_memory_length},max_frames_num=${max_frames_num},force_sample=False,video_fps=${fps},torch_dtype=bfloat16,min_frames_num=${min_frames_num},set_verbose=True"

#----- Task settings -----

model="qvic_mf"
tasks="mlvu_test"
#tasks="mlvu_dev"
#tasks="longvideobench_val_v"
#tasks="videomme"

#-----

PREV_STAGE_CHECKPOINT_CLEAN="${pretrained//\//_}"
MID_RUN_NAME=${PREV_STAGE_CHECKPOINT_CLEAN}_${tasks}
datetime="$(date +%m%d%H%M%S)"
LOG_DIR="./logs/eval/${MID_RUN_NAME}"
LOG_FILE="${LOG_DIR}/${datetime}.log"
mkdir -p "${LOG_DIR}"

echo "pretrained: ${pretrained}"
echo "parameter: ${parameter}"
echo "model: ${model}"
echo "tasks: ${tasks}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
echo "log: ${LOG_FILE}"

trap 'echo "[ERROR] eval failed. See log: ${LOG_FILE}" >&2' ERR

accelerate launch --num_processes=8 \
    -m lmms_eval \
    --model "${model}" \
    --model_args "${parameter}" \
    --tasks "${tasks}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qvic_mf \
    --output_path ./work_dirs/lmms-eval/ \
    2>&1 | tee -a "${LOG_FILE}"
