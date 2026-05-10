import json
import os
from itertools import combinations


def convert_to_pairwise_json(input_file, output_file, group_size=16):
    """
    Convert model outputs from the same source text into pairwise comparison format.

    Parameters:
        input_file (str): Path to the input JSONL file containing model annotations.
        output_file (str): Path to write the output JSON file.
        group_size (int): Number of records per group, all sharing the same source text.

    Returns:
        list: A list of pairwise comparison dictionaries.
    """
    
    all_records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))
    
    groups = []
    for i in range(0, len(all_records), group_size):
        group = all_records[i:i+group_size]
        if len(group) == group_size:
            texts = [r['text'] for r in group]
            if all(t == texts[0] for t in texts):
                groups.append(group)
    
    all_pairs = []
    
    for group_idx, group in enumerate(groups):
        for i, j in combinations(range(len(group)), 2):
            rec_a = group[i]
            rec_b = group[j]
            
            judges = []
            winners = []
            
            for idx, (ann_a, ann_b) in enumerate(zip(rec_a['expert_annotations'], rec_b['expert_annotations'])):
                score_a = sum(ann_a.values()) / len(ann_a)
                score_b = sum(ann_b.values()) / len(ann_b)
                
                if abs(score_a - score_b) < 1e-6:
                    winner = 'tie'
                elif score_a > score_b:
                    winner = 'model_a'
                else:
                    winner = 'model_b'
                
                judges.append(f"expert_{idx+1}")
                winners.append(winner)
            
            for idx, (ann_a, ann_b) in enumerate(zip(rec_a['turker_annotations'], rec_b['turker_annotations'])):
                score_a = sum(ann_a.values()) / len(ann_a)
                score_b = sum(ann_b.values()) / len(ann_b)
                
                if abs(score_a - score_b) < 1e-6:
                    winner = 'tie'
                elif score_a > score_b:
                    winner = 'model_a'
                else:
                    winner = 'model_b'
                
                judges.append(f"turker_{idx+1}")
                winners.append(winner)
            
            pair = {
                "unique_id": f"eval_{group_idx:06d}_{rec_a['model_id']}_vs_{rec_b['model_id']}",
                "question_id": rec_a['id'],
                "model_a": rec_a['model_id'],
                "model_b": rec_b['model_id'],
                "text": rec_a['text'],
                "model_a_output": rec_a['decoded'],
                "model_b_output": rec_b['decoded'],
                "judges": judges,
                "winners": winners,
                "score_details": {
                    "model_a": {
                        "expert": rec_a['expert_annotations'],
                        "turker": rec_a['turker_annotations']
                    },
                    "model_b": {
                        "expert": rec_b['expert_annotations'],
                        "turker": rec_b['turker_annotations']
                    }
                }
            }
            all_pairs.append(pair)
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    
    return all_pairs


if __name__ == "__main__":
    convert_to_pairwise_json(
        input_file="model_annotations.aligned.paired.jsonl",
        output_file="summeval_pairwise.json",
        group_size=16
    )
