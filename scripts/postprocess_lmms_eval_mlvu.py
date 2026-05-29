import os
import re

def main(args):
    """
    Convert a JSONL file to a CSV file.
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    import pandas as pd
    import json

    with open(args.json, 'r') as f:
        json_data = json.load(f)
        json_task = list(json_data['configs'].keys())[0]
        json_model_name = json_data['model_name_sanitized']
        if 'mlvu_percetion_score,none' in json_data['results'][json_task]:
            json_score = json_data['results'][json_task]['mlvu_percetion_score,none']
            score_name = 'mlvu_percetion_score'
        else:
            print("No score found in the JSON data.")
            breakpoint()
        json_score = f"{json_score:.2f}"
    
    # Read the JSONL file
    jsonl_file = args.json.replace('results', f'samples_{json_task}')
    jsonl_file = jsonl_file.replace('.json', '.jsonl')
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()

    data = []

    if json_task == "mlvu_test":
        TASK_TYPES = ["topic_reasoning", "anomaly_reco", "needleQA", "ego", "plotQA", "sportsQA", "order", "count", "tutorialQA"] # leaderboard order
    elif json_task == "mlvu_dev":
        TASK_TYPES = ["topic_reasoning", "anomaly_reco", "needle", "ego", "plotQA", "order", "count"]
    else:
        print(f"Unknown task type: {json_task}. Please check the task type.")
        breakpoint()
    
    category2score_orginal = {task_type: {"correct": 0, "answered": 0} for task_type in TASK_TYPES} # lmms-eval original
    category2score_refined = {task_type: {"correct": 0, "answered": 0} for task_type in TASK_TYPES} # postprocess refined
    
    for line in lines:
        entry = json.loads(line)
        doc = entry['doc']
        task_type = doc['task_type']
        pred_answer_orginal = entry[score_name]['pred_answer']
        gt_answer = doc['answer']
        
        category2score_orginal[task_type]["answered"] += 1
        category2score_orginal[task_type]["correct"] += pred_answer_orginal == gt_answer 
        
        # refine the pred_answer_orginal
        # ref. ReTaKe https://github.com/SCZwangxiao/video-ReTaKe/blob/main/retake/infer_eval.py#L25
        pred_answer_refined = pred_answer_orginal
        matches = re.search(r"[ABCDEFG]", pred_answer_orginal) 
        if matches:
            pred_answer_refined = matches.group(0)
        
        category2score_refined[task_type]["answered"] += 1
        category2score_refined[task_type]["correct"] += pred_answer_refined == gt_answer
        
    task_category_scores_original = {}
    task_category_scores_refined = {}
    
    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score_orginal.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores_original[task_cate] = accuracy
        print(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")
        
    average_accuracy = sum(task_category_scores_original.values()) / len(TASK_TYPES)
    print(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")
    
    # Calculate and log accuracy for each task category (refined)
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score_refined.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores_refined[task_cate] = accuracy
        print(f"Refined Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")
        
    average_accuracy_refined = sum(task_category_scores_refined.values()) / len(TASK_TYPES)
    print(f"Refined Average Performance Across All Task Categories: {average_accuracy_refined:.1f}%")
    
    # Save csv
    # columns: average, topic_reasoning, anomaly_reco, needleQA, ego, plotQA, sportsQA, order, count", tutorialQA
    # rows: original_correct, original_answered, original_accuracy, refined_correct, refined_answered, refined_accuracy
    # -> csv_file: outdir/refined_{task}_{score:%f.2}_to_{score:%f.2}_{model_name}_results.csv
    output_file = f"{args.outdir}/refined_{json_task}_{json_score}_to_{average_accuracy_refined:.2f}_{json_model_name}_results.csv"
    os.makedirs(args.outdir, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("average,")
        f.write(",".join(TASK_TYPES) + "\n")
        f.write(f"{average_accuracy:.2f},")
        f.write(",".join(f"{task_category_scores_original[task_cate]:.2f}" for task_cate in TASK_TYPES) + "\n")
        f.write(f"{average_accuracy_refined:.2f},")
        f.write(",".join(f"{task_category_scores_refined[task_cate]:.2f}" for task_cate in TASK_TYPES) + "\n")

        
    print(f"Results saved to {output_file}")
    
    
if __name__ == "__main__":
    import argparse
    argparse = argparse.ArgumentParser(description="Process JSON file and convert to CSV.")
    argparse.add_argument('--json', type=str, # *_results.json
                        required=True, help='Path to the input JSONL file.')
    argparse.add_argument('--outdir', type=str, default="scripts/log/", help='Path to the output directory.')
    args = argparse.parse_args()
    main(args)
