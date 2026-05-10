"""
Data loading and conversion.
"""

import json
from typing import List, Dict, Tuple
from .data_types import SampleData


def load_mt_bench_data(filepath: str, max_samples: int = None) -> List[dict]:
    """
    Load raw MT Bench data from a file.

    Args:
        filepath: Path to the data file.
        max_samples: Maximum number of samples to load (for debugging).

    Returns:
        List of raw data items.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    return data


def convert_mt_bench_to_samples(data_items: List[Dict]) -> List[SampleData]:
    """
    Convert MT Bench format data into evaluation samples.

    IMPORTANT: All samples are assumed to be annotated by the same two dummy
    raters (rater_0, rater_1). Actual rater IDs in the dataset are ignored so
    that the bias matrices for these two fixed raters can be learned jointly.

    Each sample is expected to have 2 annotations (the ``winners`` list has
    length 2), corresponding to rater_0 and rater_1 respectively.

    Args:
        data_items: List of raw MT Bench data items.

    Returns:
        List of SampleData objects.
    """
    samples = []
    label_map = {'model_a': 'A', 'model_b': 'B', 'tie': 'T'}

    for item in data_items:
        conversation_a = item.get("conversation_a", [])
        conversation_b = item.get("conversation_b", [])

        human_judgments = {}
        if 'winners' in item and isinstance(item['winners'], list):
            for idx, winner in enumerate(item['winners'][:2]):
                if winner in label_map:
                    rater_id = f"rater_{idx}"
                    human_judgments[rater_id] = label_map[winner]

        llm_judgments = item.get('llm_swap_judgments', {})

        metadata = {
            'question_id': item.get('question_id'),
            'model_a': item.get('model_a'),
            'model_b': item.get('model_b'),
            'turn': item.get('turn')
        }

        samples.append(SampleData(
            human_judgments=human_judgments,
            llm_judgments=llm_judgments,
            metadata=metadata,
            conversation_a=conversation_a,
            conversation_b=conversation_b
        ))

    return samples


def load_reward_bench_pairs(filepath: str, max_samples: int = None) -> List[Dict]:
    """
    Load converted Reward Bench pair data.

    Args:
        filepath: Path to the reward_bench_pairs_openai.json file.
        max_samples: Maximum number of samples to load.

    Returns:
        List of pair data dictionaries.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    return data


def convert_reward_bench_to_samples(
    pairs: List[Dict],
    use_ground_truth: bool = True
) -> List[SampleData]:
    """
    Convert Reward Bench pair data into evaluation samples.

    Reward Bench data structure:
    - chosen: {messages: [{role: user/assistant, content: ...}]}
    - rejected: {messages: [{role: user/assistant, content: ...}]}
    - metadata: original metadata

    Conversion strategy:
    - chosen -> conversation_a (better response)
    - rejected -> conversation_b (worse response)
    - If ``use_ground_truth`` is True: human label is 'A' (chosen is better).

    Args:
        pairs: List of Reward Bench pairs.
        use_ground_truth: Whether to use ground truth as the human label
            (chosen is better).

    Returns:
        List of SampleData objects.
    """
    samples = []

    for idx, pair in enumerate(pairs):
        chosen_messages = pair.get("chosen", {}).get("messages", [])
        rejected_messages = pair.get("rejected", {}).get("messages", [])

        if not chosen_messages or not rejected_messages:
            continue

        chosen_conv = chosen_messages if len(chosen_messages) > 1 else [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": chosen_messages[0].get("content", "")}
        ]

        rejected_conv = rejected_messages if len(rejected_messages) > 1 else [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": rejected_messages[0].get("content", "")}
        ]

        human_judgments = {}
        if use_ground_truth:
            human_judgments["rater_0"] = "A"
            human_judgments["rater_1"] = "A"

        llm_judgments = pair.get("llm_swap_judgments", {})

        metadata = pair.get("metadata", {})
        metadata["pair_id"] = idx
        metadata["dataset"] = "reward_bench"

        samples.append(SampleData(
            human_judgments=human_judgments,
            llm_judgments=llm_judgments,
            metadata=metadata,
            conversation_a=chosen_conv,
            conversation_b=rejected_conv
        ))

    return samples


def format_reward_bench_for_display(conversation: List[Dict[str, str]]) -> str:
    """
    Format a Reward Bench conversation as a display string.

    Args:
        conversation: List of messages in OpenAI format.

    Returns:
        Formatted string.
    """
    lines = []
    for msg in conversation:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]\n{content}\n")
    return "\n".join(lines)
