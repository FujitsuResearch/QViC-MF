"""Aggregate per-sample lmms-eval output into a CSV for manual inspection.

Given an lmms-eval ``*_results.json`` file, this script locates the matching
``*_samples_<task>.jsonl`` (per-sample dump) in the same directory and emits a
CSV with one row per QA: doc_id, task_type, video_name, duration, input
prompt, raw response, predicted answer, ground-truth target, and whether the
prediction matches.

Primarily intended for MLVU (``mlvu_test`` / ``mlvu_dev``) but also handles
``nextqa_mc_test``, ``activitynetqa``, and ``llava_in_the_wild`` outputs.

Example
-------
    python scripts/analyze_mlvu_lmms_eval_results.py \\
        --json work_dirs/lmms-eval/<run_dir>/<timestamp>_results.json \\
        --outdir scripts/log/

The output CSV is written to
``<outdir>/<task>_<score>_<model_name>_results.csv``.
"""

import os

def main(args):
    """
    Convert a JSONL file to a CSV file.
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    import pandas as pd
    import json

    # Extract task ("mlvu_test"), model_args, and score (e.g. "12.44") from args.json.
    with open(args.json, 'r') as f:
        json_data = json.load(f)
        json_task = list(json_data['configs'].keys())[0]
        json_model_name = json_data['model_name_sanitized']
        if 'mlvu_percetion_score,none' in json_data['results'][json_task]:
            json_score = json_data['results'][json_task]['mlvu_percetion_score,none']
            score_name = 'mlvu_percetion_score'
        elif 'gpt_eval_accuracy,none' in json_data['results'][json_task]:
            json_score = json_data['results'][json_task]['gpt_eval_accuracy,none']
            score_name = 'gpt_eval_accuracy'
        elif 'nextqa_mc_test' in json_data['results']:
            json_score = json_data['results']['nextqa_mc_test']["exact_match,none"]
            score_name = 'nextqa_mc_test'
        elif 'gpt_eval_llava_all,none' in json_data['results'][json_task]:
            json_score = json_data['results'][json_task]["gpt_eval_llava_all,none"]
            score_name = 'gpt_eval_llava_all'
        else:
            print("No score found in the JSON data.")
            breakpoint()

        json_score = f"{json_score:.2f}"
    
    # Read the JSONL file
    # args.json: work_dirs/lmms-eval/iqvic-next__iqvic-next_liuhaotian_llava-v1.5-7b_to_LLaVA-Finetune_ctx64_all-linear_addAssistant_splitQA_EncVid/20250407_194233_results.jsonl
    # jsonl_file: work_dirs/lmms-eval/iqvic-next__iqvic-next_liuhaotian_llava-v1.5-7b_to_LLaVA-Finetune_ctx64_all-linear_addAssistant_splitQA_EncVid/20250407_194233_samples_mlvu_test.json
    jsonl_file = args.json.replace('results', f'samples_{json_task}')
    jsonl_file = jsonl_file.replace('.json', '.jsonl')
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()

    # Initialize a list to store the data
    data = []

    # Process each line in the JSONL file
    for line in lines:
        entry = json.loads(line)
        doc = entry['doc']
        doc_id = entry['doc_id']
        if 'task_type' in doc:
            task_type = doc['task_type']
        elif 'type' in doc:
            task_type = doc['type']
        elif 'type' in entry[score_name]:
            task_type = entry[score_name]['type']
        else:
            task_type = None
        if 'video_name' in doc:
            video_name = doc['video_name']
        elif 'video' in doc:
            video_name = doc['video']
        else:
            video_name = None
        duration = doc['duration'] if 'duration' in doc else None
        input_text = entry['input']
        resps = entry['resps']
        #filtered_resps = entry['filtered_resps']
        if 'answer' in doc:
            answer = doc['answer']
        elif 'answer' in entry[score_name]:
            answer = entry[score_name]['answer']
        target = entry['target']
        if 'pred_answer' in entry[score_name]:
            pred_answer = entry[score_name]['pred_answer']
        elif 'filtered_resps' in entry:
            pred_answer = entry['filtered_resps']
        elif 'pred' in entry[score_name]:
            pred_answer = entry[score_name]['pred']
        else:
            pred_answer = None

        if json_task == 'mlvu_test':
            match = pred_answer == answer
        elif json_task == 'activitynetqa':
            match = entry[score_name]['Correctness'].lower() == 'yes'
        elif json_task == 'nextqa_mc_test':
            match = entry['exact_match']
        elif json_task == 'llava_in_the_wild':
            match = f"ref(gpt): {entry[score_name]['scores'][0]}, pred: {entry[score_name]['scores'][1]}"
        else:
            match = None
        
        # Escape any newlines in the input text and ground-truth target so the CSV stays single-line per row.
        input_text = input_text.replace('\n', '\\n')
        target = target.replace('\n', '\\n')

        # Append the data to the list
        data.append({
            'doc_id': doc_id,
            'task_type': task_type,
            'video_name': video_name,
            'duration': duration,
            'input': input_text,
            'resps': resps,
            'pred_answer': pred_answer,
            'target': target,
            'match': match
        })

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    # sort the DataFrame by doc_id
    df.sort_values(by='doc_id', inplace=True)
    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Save the DataFrame to a CSV file
    # -> csv_file: outdir/{task}_{score:%f.2}_{model_name}_results.csv
    output_file = f"{args.outdir}/{json_task}_{json_score}_{json_model_name}_results.csv"
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Dump per-sample QA from an lmms-eval result file into a CSV "
            "(one row per QA: input, response, prediction, target, match). "
            "Primarily for MLVU; also handles nextqa_mc_test / activitynetqa / "
            "llava_in_the_wild."
        ),
        epilog=(
            "Example:\n"
            "  python scripts/analyze_mlvu_lmms_eval_results.py \\\n"
            "      --json work_dirs/lmms-eval/<run_dir>/<timestamp>_results.json \\\n"
            "      --outdir scripts/log/\n\n"
            "Output: <outdir>/<task>_<score>_<model_name>_results.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help=(
            "Path to an lmms-eval *_results.json file. The matching "
            "*_samples_<task>.jsonl in the same directory is read automatically."
        ),
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default="scripts/log/",
        help="Directory to write the output CSV into (default: %(default)s).",
    )
    args = parser.parse_args()
    main(args)
