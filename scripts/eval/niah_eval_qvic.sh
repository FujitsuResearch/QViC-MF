#!/usr/bin/env bash
set -euo pipefail

export NCCL_NVLS_ENABLE=0

if ! python -c "import qvic" >/dev/null 2>&1; then
    echo "[ERROR] 'qvic' is not importable from $(command -v python)." >&2
    echo "        Activate the venv first:  source .venv/bin/activate" >&2
    echo "        Or install the package:   pip install -e ." >&2
    exit 1
fi

ANNOFILE="<path to VNBench-main-4try.json>"
VIDEODIR="<path to video folder>"
OUTPUT="./logs/eval/vnbench/"

pretrained="ckpt/QViC-MF-7B"  # TODO: switch back to "Fujitsu/QViC-MF-7B" once published on HF

model_base="lmms-lab/LLaVA-Video-7B-Qwen2"

FRAMES=-1
FPS=2
CONTEXT_MEMORY_LENGTH=256 
compress_with_relevance=true
context_condition_frame_num=32

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

if [[ ! -f "${ANNOFILE}" ]]; then
    echo "[ERROR] ANNOFILE not found: ${ANNOFILE}" >&2
    echo "        Edit ANNOFILE in $(basename "$0") to point to VNBench-main-4try.json." >&2
    exit 1
fi
if [[ ! -d "${VIDEODIR}" ]]; then
    echo "[ERROR] VIDEODIR not found: ${VIDEODIR}" >&2
    echo "        Edit VIDEODIR in $(basename "$0") to point to the VNBench video folder." >&2
    exit 1
fi

#---

PREV_STAGE_CHECKPOINT_CLEAN="${pretrained//\//_}"
MID_RUN_NAME=${PREV_STAGE_CHECKPOINT_CLEAN}_${FRAMES}frm_${FPS}fps
datetime="$(date +%m%d%H%M%S)"
OUTPUT=logs/eval_vnbench/${MID_RUN_NAME}/

export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPUs"

RUN_DIR="${OUTPUT}/video_niah_${datetime}"
LOG_FILE="${RUN_DIR}/stdout.log"
mkdir -p "${RUN_DIR}"
echo "log: ${LOG_FILE}"

trap 'echo "[ERROR] eval failed. See log: ${LOG_FILE}" >&2' ERR

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_ID=${GPULIST[$IDX]}  # Note: Zsh arrays are 1-indexed by default
    echo "Running on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 qvic/eval/model_video_niah.py \
    --model-path "$pretrained" \
    --model-base "$model_base" \
    --video_dir "$VIDEODIR" \
    --question_fp "$ANNOFILE" \
    --output_dir "${RUN_DIR}" \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $(($IDX - 1)) \
    --frames_num $FRAMES \
    --video_fps $FPS \
    --context_memory_length $CONTEXT_MEMORY_LENGTH \
    --compress_with_relevance $compress_with_relevance \
    --context_condition_frame_num $context_condition_frame_num \
    >> "${LOG_FILE}" 2>&1 &

done
wait
output_file="${RUN_DIR}/merge.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq -1 $((CHUNKS-2))); do
    cat "${RUN_DIR}/${CHUNKS}_${IDX}.json" >> "$output_file"
done

outdir=scripts/log/vnbench/${MID_RUN_NAME}_eval_${datetime}/
echo "./scripts/video/eval/evaluation_utils.py --annotation_path $ANNOFILE --result_path $output_file --outdir $outdir"

python ./scripts/video/eval/evaluation_utils.py --annotation_path "$ANNOFILE" --result_path "$output_file" \
    --outdir "$outdir" \
    2>&1 | tee -a "${LOG_FILE}"
