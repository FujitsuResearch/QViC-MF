# Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2603.15167-b31b1b.svg)](https://arxiv.org/abs/2603.15167)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Fujitsu%2FQViC--MF--7B-yellow)](https://huggingface.co/Fujitsu/QViC-MF-7B)

The official repository for the paper **Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding** (CVPR 2026).

Sosuke Yamao\*, Natsuki Miyahara\*, Yuankai Qi, Shun Takeuchi (Fujitsu Research / Macquarie University)

\*Equal contribution

<!--
<p align="center">
  <img src="assets/fig1.png" width="360">
  &nbsp;&nbsp;
  <img src="assets/fig2.png" width="690">
</p>
-->
<p align="center">
  <img src="assets/fig2.png" width="690">
</p>

## News

- **[2026.03]** Paper accepted to CVPR 2026. [[arXiv]](https://arxiv.org/abs/2603.15167)
- **[2026.05]** Code and pretrained weights ([`Fujitsu/QViC-MF-7B`](https://huggingface.co/Fujitsu/QViC-MF-7B)) released.

## Abstract

In the context of long-term video understanding with large multimodal models, many frameworks have been proposed. Although transformer-based visual compressors and memory-augmented approaches are often used to process long videos, they usually compress each frame independently and therefore fail to achieve strong performance on tasks that require understanding complete events, such as temporal ordering tasks in MLVU and VNBench. This motivates us to rethink the conventional one-way scheme from perception to memory, and instead establish a feedback-driven process in which past visual contexts stored in the context memory can benefit ongoing perception. To this end, we propose **Question-guided Visual Compression with Memory Feedback (QViC-MF)**, a framework for long-term video understanding. At its core is a **Question-guided Multimodal Selective Attention (QMSA)**, which learns to preserve visual information related to the given question from both the current clip and the past related frames from the memory. The compressor and memory feedback work iteratively for each clip of the entire video. This simple yet effective design yields large performance gains on long-term video understanding tasks. Extensive experiments show that our method achieves significant improvement over current state-of-the-art methods by **6.1%** on MLVU test, **8.3%** on LVBench, **18.3%** on VNBench Long, and **3.7%** on VideoMME Long.

## Repository contents

- `qvic/` &mdash; the QViC Python package (model definition, conversation templates, multimodal utilities, standalone LVBench / VNBench evaluators).
- `lmms-eval/` &mdash; a fork of [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) that registers QViC as an evaluable model (`qvic_mf`).
- `playground/demo/inference_qvic.py` &mdash; a small interactive inference demo.
- `scripts/eval/` &mdash; shell wrappers for the evaluation pipelines.
- `scripts/analyze_*.py`, `scripts/postprocess_*.py`, `scripts/lvbench_format_annotation.py` &mdash; helpers for aggregating and analyzing evaluation results.

Training code is **not** included in this release.

## Table of contents

- [Installation](#installation)
- [Download the model](#download-the-model)
- [Quick inference demo](#quick-inference-demo)
- [Evaluation with lmms-eval](#evaluation-with-lmms-eval)
- [Evaluation on LVBench and VNBench](#evaluation-on-lvbench-and-vnbench)
- [Result analysis utilities](#result-analysis-utilities)
- [Citation](#citation)
- [License](#license)

## Installation

We recommend Python 3.10 and CUDA 12.1 with PyTorch 2.5.1.

```bash
# Use pyenv (or any Python 3.10 installation)
pyenv install 3.10.14
pyenv local 3.10.14

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip  # Enable PEP 660 support.

# PyTorch (CUDA 12.1 build) -- install first, before any other package
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# lmms-eval (installed as editable so the bundled QViC adapter is picked up)
git submodule update --init --recursive
cd lmms-eval
pip install -e .
cd ..

# QViC package and runtime dependencies
# (re-pins transformers / tokenizers / peft to the versions QViC needs)
pip install -e .[eval]

# Optional: FlashAttention 2 for faster inference
pip install flash-attn --no-build-isolation

# Optional: extra packages for debugging / visualization
pip install plotly seaborn
```

Some benchmarks rely on a Hugging Face token and/or OpenAI key for evaluation and grading:

```bash
export HF_TOKEN=...
export OPENAI_API_KEY=...

# Optional: a shared Hugging Face cache directory
export HF_HOME=/path/to/hf_cache
```

Additional notes:

- Pointing `HF_HOME` to a shared cache directory avoids re-downloading models and datasets across multiple evaluation jobs.
- If you share Hugging Face cache data between users, remember to set the appropriate filesystem permissions so that the other users can read the cached files.
- Installing `flash-attn` is optional; the model also runs with `attn_implementation="sdpa"`.

## Download the model

The released checkpoint is a LoRA-style adapter on top of `lmms-lab/LLaVA-Video-7B-Qwen2`. Both the base model and the adapter are loaded automatically by `qvic.model.builder.load_pretrained_model`, but they need to be available either through the local Hugging Face cache or directly downloadable.

```bash
# Cache the adapter weights
huggingface-cli download Fujitsu/QViC-MF-7B

# Cache the base model
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2
```

When passing `pretrained="Fujitsu/QViC-MF-7B"` and `model_base="lmms-lab/LLaVA-Video-7B-Qwen2"`, the loader resolves both repositories from the cache.

## Quick inference demo

`playground/demo/inference_qvic.py` is a tiny interactive REPL that loads QViC, samples 64 frames from a short sample video (`playground/demo/xU25MMA2N4aVtYay.mp4`, "SORA" written by clouds, ~3 seconds), and answers any query you type.

```bash
CUDA_VISIBLE_DEVICES=0 python playground/demo/inference_qvic.py
```

Example: enter `Please describe this video in detail.` at the prompt.

```
Model Class: LlavaQwenForCausalLM
context_memory_length: 256 (updated)
Video loaded: playground/demo/xU25MMA2N4aVtYay.mp4
Video time: 3.20 seconds
Number of frames: 64
video[0].shape: torch.Size([64, 3, 384, 384])

 **Input query**

Please describe this video in detail.
Question:
 <image>
Please describe this video in detail.
Answer:
 The video begins with a serene view of a clear blue sky, where a few small white clouds are scattered. The camera focuses on a larger cloud formation in the center, which gradually takes shape and reveals the word 'SORA' written in large, white, fluffy letters. As the cloud continues ...
```

VRAM footprint: approximately 35 GB on a single H200 / A100 80 GB GPU.

### Tips

- Type `b` instead of a question to drop into a `breakpoint()`. You can then mutate variables interactively and `c` to continue (handy when tuning frame sampling, memory length, etc.).
- `context_memory_length` (default `256`) sets the number of tokens kept in the context memory. After changing it from a breakpoint, also call `model.set_context_memory_length(context_memory_length)`.
- `model.set_debug_drawing_memory(True)` saves per-step images and relevance-score plots to `./logs/context_memory/images/`.
- Setting `max_frames_num = -1` switches to FPS sampling. The sampling rate is controlled by `fps`.
- To swap the input video without reloading the model, change `video_path` and call `video, image_sizes, modalities, video_time, frame_time, frame_idx = load_video_tensor(video_path, max_frames_num)` again.

## Evaluation with lmms-eval

QViC is registered as a model in the bundled lmms-eval fork:

- `qvic_mf` &mdash; video benchmarks

The scripts under `scripts/eval/` wrap a typical `accelerate launch -m lmms_eval ...` invocation.

```bash
# Run the default configuration (MLVU test, qvic_mf, 8 GPUs, fps=2)
bash scripts/eval/lmms_eval_qvic.sh
```

The script writes per-sample results under `./work_dirs/lmms-eval/` and logs under `./logs/eval/`.

### Switching benchmarks

Edit the `tasks` variable in `scripts/eval/lmms_eval_qvic.sh` to one of:

Long-form video understanding (use `model="qvic_mf"`):

| Task | Description |
| --- | --- |
| `mlvu_test` | MLVU (test split) |
| `mlvu_dev` | MLVU (dev split) |
| `videomme` | Video-MME |

### Post-processing MLVU results

For both `mlvu_test` and `mlvu_dev`, run `scripts/postprocess_lmms_eval_mlvu.py` on the lmms-eval result file to get the refined per-category accuracy (the script strips trailing punctuation from the predicted answers before scoring, which matches the official leaderboard protocol):

```bash
python scripts/postprocess_lmms_eval_mlvu.py \
    --outdir scripts/log/ \
    --json work_dirs/lmms-eval/<run_dir>/<timestamp>_results.json
```

The script writes a CSV (`refined_<task>_<original>_to_<refined>_<model>_results.csv`) containing both the original and refined accuracy per task category.

For an item-level view of each QA (input prompt, raw response, predicted answer, ground truth, match), pass the same `--json` to `scripts/analyze_mlvu_lmms_eval_results.py`:

```bash
python scripts/analyze_mlvu_lmms_eval_results.py \
    --json work_dirs/lmms-eval/<run_dir>/<timestamp>_results.json \
    --outdir scripts/log/
```

Arguments:

- `--json` &mdash; **required.** Path to the lmms-eval `*_results.json`. The matching `*_samples_<task>.jsonl` in the same directory is read automatically.
- `--outdir` &mdash; output directory (default: `scripts/log/`). The CSV is written to `<outdir>/<task>_<score>_<model_name>_results.csv`.

## Evaluation on LVBench and VNBench

For LVBench and VNBench we provide standalone scripts (not routed through lmms-eval) under `qvic/eval/`:

- `qvic/eval/model_video_lvbench.py`
- `qvic/eval/model_video_niah.py` (used for VNBench)

The shell wrappers expect the path placeholders to be filled in:

```bash
# LVBench
bash scripts/eval/lvbench_eval_qvic.sh
# VNBench
bash scripts/eval/niah_eval_qvic.sh
```

Before running, edit the following variables at the top of each script:

- `ANNOFILE` / `ANNOFILE_FORMAT` &mdash; paths to the benchmark annotation files.
- `VIDEODIR` &mdash; directory containing the benchmark videos.
- `pretrained` &mdash; defaults to `Fujitsu/QViC-MF-7B`.
- `model_base` &mdash; defaults to `lmms-lab/LLaVA-Video-7B-Qwen2`.

For LVBench, `scripts/lvbench_format_annotation.py` converts the original `video_info.meta.jsonl` into the formatted variant expected by `qvic/eval/model_video_lvbench.py`.

## Result analysis utilities

A few small helpers are kept under `scripts/`:

- `scripts/postprocess_lmms_eval_mlvu.py` &mdash; re-aggregates MLVU accuracy after stripping trailing punctuation, outputting a CSV summary. Should be run on every `mlvu_test` / `mlvu_dev` result (see [Post-processing MLVU results](#post-processing-mlvu-results)).
- `scripts/analyze_mlvu_lmms_eval_results.py` &mdash; dumps per-sample QA into CSV for manual inspection (primarily MLVU; also handles `nextqa_mc_test` / `activitynetqa` / `llava_in_the_wild`). See [Post-processing MLVU results](#post-processing-mlvu-results) for usage.
- `scripts/lvbench_format_annotation.py` &mdash; converts an LVBench `video_info.meta.jsonl` into the formatted variant consumed by `qvic/eval/model_video_lvbench.py`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yamao2026qvicmf,
  title={Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding},
  author={Yamao, Sosuke and Miyahara, Natsuki and Qi, Yuankai and Takeuchi, Shun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

The code in this repository is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA-4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. See [`LICENSE`](LICENSE) for details.
The bundled `lmms-eval/` directory retains its original license; see [`lmms-eval/LICENSE`](lmms-eval/LICENSE).

The released checkpoint [`Fujitsu/QViC-MF-7B`](https://huggingface.co/Fujitsu/QViC-MF-7B) is distributed under the [CC-BY-NC-ND-4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license. It is fine-tuned from `lmms-lab/LLaVA-Video-7B-Qwen2`, which is itself trained on the LLaVA-Video-178K and LLaVA-OneVision datasets; please refer to the upstream license terms for any downstream use of the weights.
