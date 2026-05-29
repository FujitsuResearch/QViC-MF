# A script to separate annotations in LVBench's video_info.meta.jsonl by sample

import json
import argparse

def format_lvbench_annotation(input_path, output_path):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            video_key = data['key']
            video_type = data['type']
            video_info = data['video_info']
            for qa_pair in data['qa']:
                formatted_entry = {
                    'key': f"{video_key}_{qa_pair['uid']}",
                    'type': video_type,
                    'question': qa_pair['question'],
                    'answer': qa_pair['answer'],
                    'question_type': qa_pair['question_type'],
                    'time_reference': qa_pair['time_reference'],
                    'video_info': video_info
                }
                f_out.write(json.dumps(formatted_entry) + '\n')
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format LVBench annotations")
    parser.add_argument('--input', type=str,  
                        required=True, #video_info.meta.jsonl"
                        help="Path to the input LVBench video_info.meta.jsonl file")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to the output formatted JSONL file")
    args = parser.parse_args()
    
    format_lvbench_annotation(args.input, args.output)