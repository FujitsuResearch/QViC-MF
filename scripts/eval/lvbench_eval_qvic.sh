
export NCCL_NVLS_ENABLE=0

ANNOFILE_FORMAT="<path to video_info.meta.formatted.jsonl>"
ANNOFILE="<path to video_info.meta.jsonl>"
VIDEODIR="<path to video folder>"
OUTPUT="./logs/eval/lvbench/"

pretrained="ckpt/QViC-MF-7B"  # TODO: switch back to "Fujitsu/QViC-MF-7B" once published on HF

model_base="lmms-lab/LLaVA-Video-7B-Qwen2"

FRAMES=2048
FPS=2
CONTEXT_MEMORY_LENGTH=256
compress_with_relevance=true
context_condition_frame_num=32

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

#---

PREV_STAGE_CHECKPOINT_CLEAN="${pretrained//\//_}"
MID_RUN_NAME=${PREV_STAGE_CHECKPOINT_CLEAN}_${FRAMES}frm_${FPS}fps
datetime="$(date +%m%d%H%M%S)"
OUTPUT=logs/eval_lvbench/${MID_RUN_NAME}/

export PYTHONPATH="./:$PYTHONPATH"
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo "Using $CHUNKS GPUs"

mkdir -p $OUTPUT/video_lvbench_${datetime}

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_ID=${GPULIST[$IDX]}  # Note: Zsh arrays are 1-indexed by default

    # If IDX is the first, set set_verbose to True
    if [ $IDX -eq 0 ]; then
        echo "Setting verbose True for chunk index $IDX"
        set_verbose=True
    else
        echo "Setting verbose False for chunk index $IDX"
        set_verbose=False
    fi

    echo "Running on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 qvic/eval/model_video_lvbench.py \
    --model-path $pretrained \
    --model-base $model_base \
    --video_dir $VIDEODIR \
    --question_fp $ANNOFILE_FORMAT \
    --output_dir $OUTPUT/video_lvbench_${datetime} \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $(($IDX - 1)) \
    --frames_num $FRAMES \
    --video_fps $FPS \
    --context_memory_length $CONTEXT_MEMORY_LENGTH \
    --compress_with_relevance $compress_with_relevance \
    --context_condition_frame_num $context_condition_frame_num \
    --set_verbose $set_verbose \
    >> $OUTPUT/video_lvbench_${datetime}/stdout.log 2>&1 &

done
wait
output_file=$OUTPUT/video_lvbench_${datetime}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

if [ $CHUNKS -eq 1 ]; then
    cp $OUTPUT/video_lvbench_${datetime}/pred.json "$output_file"
else
    # Loop through the indices and concatenate each file.
    for IDX in $(seq -1 $((CHUNKS-2))); do
        cat $OUTPUT/video_lvbench_${datetime}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi

outdir=scripts/log/lvbench/${MID_RUN_NAME}_eval_${datetime}/
echo "./scripts/video/eval/evaluation_utils_lvbench.py --video_meta_file "$ANNOFILE" --answer_file "$output_file" --outdir "$outdir

python ./scripts/video/eval/evaluation_utils_lvbench.py --video_meta_file $ANNOFILE --answer_file $output_file --outdir $outdir \
    >> $OUTPUT/video_lvbench_${datetime}/stdout.log 2>&1 &
