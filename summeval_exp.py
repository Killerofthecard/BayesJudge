#!/usr/bin/env python3
"""
SummEval Bayesian Evaluation - Experiment Entry Point

Compatible with the summeval_pairwise.json dataset
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime
from pathlib import Path


class Tee:
    """Wrapper that writes output to both the console and a log file simultaneously."""
    def __init__(self, log_path: str):
        self.console = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.log_path = log_path

    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()

    def isatty(self):
        return self.console.isatty()

    def __getattr__(self, name):
        return getattr(self.console, name)

from bayesian_eval.data.data_types import SampleData, LabelMapping
from bayesian_eval.core.evaluator import StrictBayesianEvaluator
from bayesian_eval.data.data_loader import (
    format_reward_bench_for_display
)
from bayesian_eval.utils.llm_client import call_llm
try:
    from bayesian_eval.visualization.variance_plotter import VarianceConvergencePlotter
    from bayesian_eval.visualization.beta_plotter import BetaConvergencePlotter
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


def load_summeval_dataset(
    json_path: str,
    max_samples: int = None,
    llm_models: List[str] = None,
    human_judges: List[str] = None
) -> List[SampleData]:
    """Load SummEval dataset from a JSON file.

    Reads precomputed LLM judgments and supports multi-model joint modeling.

    Args:
        json_path: Path to the JSON file.
        max_samples: Maximum number of samples to load (None = all).
        llm_models: LLM model names for joint modeling.
        human_judges: Human judge IDs to filter by.

    Returns:
        List of SampleData objects.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    if max_samples is not None:
        data_list = data_list[:max_samples]
    
    samples = []
    missing_llm_counts = {m: 0 for m in (llm_models or [])}
    for item in data_list:
        winners = item.get('winners', [])
        judges = item.get('judges', [])
        
        human_judgments = {}
        for judge_id, winner in zip(judges, winners):
            if human_judges is not None and judge_id not in human_judges:
                continue
            winner = winner.lower()
            if winner == 'model_a':
                human_judgments[judge_id] = 'A'
            elif winner == 'model_b':
                human_judgments[judge_id] = 'B'
            else:
                human_judgments[judge_id] = 'T'
        
        llm_judgments = {}
        if llm_models is not None:
            item_llm_judgments = item.get('llm_judgments', {})
            for model_name in llm_models:
                if model_name in item_llm_judgments:
                    llm_data = item_llm_judgments[model_name]
                    llm_judgments[model_name] = (llm_data['label_AB'], llm_data['label_BA'])
                else:
                    missing_llm_counts[model_name] += 1
        
        conversation_a = [{"role": "assistant", "content": item.get('model_a_output', '')}]
        conversation_b = [{"role": "assistant", "content": item.get('model_b_output', '')}]
        
        sample = SampleData(
            human_judgments=human_judgments,
            llm_judgments=llm_judgments,
            metadata={
                'unique_id': item.get('unique_id'),
                'question_id': item.get('question_id'),
                'model_a': item.get('model_a'),
                'model_b': item.get('model_b'),
                'text': item.get('text'),
            },
            conversation_a=conversation_a,
            conversation_b=conversation_b
        )
        samples.append(sample)
    
    print(f"[Data Loading] Successfully loaded {len(samples)} samples from {json_path}")
    if human_judges is not None:
        print(f"[Data Loading] Specified human judges: {', '.join(human_judges)}")
    if llm_models is not None:
        print(f"[Data Loading] LLM models for joint modeling: {', '.join(llm_models)}")
        for model_name, count in missing_llm_counts.items():
            if count > 0:
                print(f"[Warning] Model '{model_name}': {count} samples missing LLM judgments")
    return samples


def parse_llm_label(response: str) -> str:
    """Parse an LLM response string into a normalized label (A, B, or T).

    Args:
        response: Raw LLM response string.

    Returns:
        Normalized label ('A', 'B', or 'T').
    """
    if not response:
        return 'T'
    resp_clean = response.strip().upper()
    if resp_clean in ['A', 'B']:
        return resp_clean
    if resp_clean in ['T', 'TIE', 'C']:
        return 'T'
    if resp_clean.startswith('A'):
        return 'A'
    if resp_clean.startswith('B'):
        return 'B'
    return 'T'


USER_PROMPT_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the summaries provided by two AI assistants to the source text displayed below.

Please only output a single letter to indicate your choice.
- Output A if Assistant A's summary is better.
- Output B if Assistant B's summary is better.
- Output C if they are tied.

**Your Output Should Be Only: A or B or C**

## Source Text
{text}

## Assistant A
{res_a}

## Assistant B
{res_b}

Which is better, A, B, or C? """


def build_swap_prompt(text: str, summary_first: str, summary_second: str) -> str:
    """Build a pairwise comparison prompt using the swap-design template.

    The LLM should answer A (first assistant is better), B (second assistant is better),
    or C (tie). In the swap design, 'A' maps to position 1 and 'B' maps to position 2.
    """
    return USER_PROMPT_TEMPLATE.format(text=text, res_a=summary_first, res_b=summary_second)


def get_live_llm_judgment(sample: SampleData, model: str) -> Tuple[str, str]:
    """Query an LLM in real time for AB and BA swap-design judgments.

    Args:
        sample: Evaluation sample containing the article text and two summaries.
        model: Name of the LLM to query.

    Returns:
        (label_AB, label_BA) where each label is 'A', 'B', or 'C'.
        Here 'A' means "choose position 1" and 'B' means "choose position 2".
    """
    text = sample.metadata.get('text', '') if sample.metadata else ''
    output_a = sample.conversation_a[0]['content'] if sample.conversation_a else ''
    output_b = sample.conversation_b[0]['content'] if sample.conversation_b else ''

    # AB order: position 1 = A, position 2 = B
    prompt_ab = build_swap_prompt(text, output_a, output_b)
    resp_ab = call_llm(prompt_ab, model)
    label_ab = parse_llm_label(resp_ab)

    # BA order: position 1 = B, position 2 = A
    prompt_ba = build_swap_prompt(text, output_b, output_a)
    resp_ba = call_llm(prompt_ba, model)
    label_ba = parse_llm_label(resp_ba)

    return label_ab, label_ba


def convert_to_native(obj):
    """Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object potentially containing numpy types.

    Returns:
        Object with all numpy types converted to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):
        return obj
    else:
        return obj


def compute_frobenius_norm_diff(matrix_learned: np.ndarray, matrix_preset: np.ndarray) -> float:
    """Compute the Frobenius norm difference between two matrices.

    Args:
        matrix_learned: Learned matrix.
        matrix_preset: Preset or target matrix.

    Returns:
        Frobenius norm of the difference.
    """
    diff = matrix_learned - matrix_preset
    return np.linalg.norm(diff, 'fro')


def print_experiment_config(config: dict):
    """Pretty-print the experiment configuration dictionary."""
    print()
    print("[Experiment Configuration]")
    print("-" * 70)
    print(f"  Experiment Name:        {config['experiment_name']}")
    print(f"  Data File:        {config['dataset_path']}")
    models = config['model']
    if isinstance(models, list):
        if len(models) == 1:
            print(f"  Eval Model:        {models[0]}")
        else:
            print(f"  Joint Models:    {', '.join(models)} ({len(models)} models)")
    else:
        print(f"  Eval Model:        {models}")
    print()
    print("  [Sampling Configuration]")
    print(f"    Samples:        {config['n_samples']}")
    print(f"    Train Samples:      {config['n_train']}")
    print(f"    Test Samples:      {config['n_test']}")
    print()
    print("  [Particle Filter Configuration]")
    print(f"    Particles:        {config['n_particles']}")
    print(f"    Lambda(Diag):  {config['lambda_diag']}")
    print(f"    Lambda(Off):   {config['lambda_off']}")
    print(f"    Sigma Beta:    {config['sigma_beta']}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="SummEval Experiment with Train/Test Split")
    parser.add_argument("--data", type=str, default="dataset/summeval_pairwise_1000.json",
                       help="SummEval data file path")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Total number of samples (default: use all data)")
    parser.add_argument("--n_train", type=int, default=800,
                       help="Number of training samples")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model name(s) with precomputed LLM judgments in the dataset, supports comma-separated multi-model joint modeling (e.g., gpt-4o-mini,gpt-4o)")
    parser.add_argument("--n_particles", type=int, default=300,
                       help="Number of particles")
    parser.add_argument("--lambda_diag", type=float, default=1.0,
                       help="Diagonal elements for human confusion matrix initialization (default: 1.0)")
    parser.add_argument("--lambda_off", type=float, default=0.0,
                       help="Off-diagonal elements for human confusion matrix initialization (default: 0.0)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: built-in outputs/ directory)")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Output file name prefix (default: built-in naming rule)")
    parser.add_argument("--no_verbose", action="store_true",
                       help="Disable verbose diagnostic printing")
    parser.add_argument("--live_plot", action="store_true",
                       help="Enable live plotting")
    parser.add_argument("--update_human", action="store_true", default=True,
                       help="Update human bias matrix C_r during training (default: True)")
    parser.add_argument("--update_llm", action="store_true", default=True,
                       help="Update LLM bias parameter beta during training (default: True)")
    parser.add_argument("--no_update_human", action="store_false", dest="update_human",
                       help="Disable updating human bias matrix during training")
    parser.add_argument("--no_update_llm", action="store_false", dest="update_llm",
                       help="Disable updating LLM bias parameter during training")
    parser.add_argument("--use_swap_logits_ba", action="store_true", default=True,
                       help="Use swapped logits_ba to compute LLM likelihood (default: True)")
    parser.add_argument("--no_swap_logits_ba", action="store_false", dest="use_swap_logits_ba",
                       help="Disable swapped logits_ba, use non-swapped version")
    parser.add_argument("--zero_human_likelihood", action="store_true",
                       help="Set human log-likelihood to 0 during test phase (only affects test inference)")
    parser.add_argument("--zero_llm_likelihood", action="store_true",
                       help="Set LLM log-likelihood to 0 during test phase (only affects test inference)")
    parser.add_argument("--human_prior", type=str, default=None,
                       help="Custom human rater prior confusion matrix JSON file path. Format: {rater_id: [[a11,a12,a13],...], ...}")
    parser.add_argument("--human_judges", type=str, default=None,
                       help="Specify participating human judges, comma-separated (e.g., expert_1,expert_2,turker_1). Default: use all")
    parser.add_argument("--live_llm", action="store_true",
                       help="Enable real-time LLM pairwise judgments during training (swap design: AB + BA)")
    parser.add_argument("--live_llm_model", type=str, default=None,
                       help="Model name for real-time LLM judgments (default: first model in --model)")

    
    args = parser.parse_args()

    args.model_list = [m.strip() for m in args.model.split(',') if m.strip()]
    
    args.human_judges_list = None
    if args.human_judges is not None:
        args.human_judges_list = [j.strip() for j in args.human_judges.split(',') if j.strip()]

    def resolve_path(p):
        if p is None:
            return None
        path = Path(p)
        if not path.is_absolute():
            path = Path(PROJECT_ROOT) / path
        return os.path.normpath(str(path))
    
    args.data = str(resolve_path(args.data))
    args.output_dir = str(resolve_path(args.output_dir)) if args.output_dir else None

    output_dir_for_log = args.output_dir if args.output_dir else os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir_for_log, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_name:
        log_name = f"{args.output_name}_{log_timestamp}.log"
    else:
        ms = datetime.now().strftime("%f")[:3]
        log_name = f"summeval_{log_timestamp}_{ms}.log"
    log_path = os.path.join(output_dir_for_log, log_name)
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"[Log] Console output also saved to: {log_path}")
    print()

    human_priors = None
    if args.human_prior is not None and args.human_prior.strip():
        prior_path = str(resolve_path(args.human_prior))
        try:
            with open(prior_path, 'r', encoding='utf-8') as f:
                prior_raw = json.load(f)
            human_priors = {}
            for rater_id, mat in prior_raw.items():
                mat_np = np.array(mat, dtype=float)
                if mat_np.shape != (3, 3):
                    print(f"[Error] Prior matrix shape for rater '{rater_id}' should be (3,3), got {mat_np.shape}")
                    return
                human_priors[rater_id] = mat_np
            print(f"[Prior Loading] Successfully loaded custom priors for {len(human_priors)} raters from {prior_path}")
        except FileNotFoundError:
            print(f"Error: Prior file not found: {prior_path}")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in prior file: {e}")
            return

    data_path = args.data
    try:
        samples = load_summeval_dataset(
            data_path,
            max_samples=args.n_samples,
            llm_models=args.model_list,
            human_judges=args.human_judges_list
        )
    except FileNotFoundError:
        print(f"Error: Data file not found: {data_path}")
        return

    n_total = len(samples)
    if args.n_train >= n_total:
        args.n_train = round(n_total * 0.8)
    n_test = n_total - args.n_train
    
    CONFIG = {
        'experiment_name': 'summeval',
        'verbose': not args.no_verbose,
        'print_interval': 1,
        'max_print_raters': 2,
        'live_plot': args.live_plot and HAS_VISUALIZATION,
        'n_samples': n_total,
        'n_train': args.n_train,
        'n_test': n_test,
        'n_particles': args.n_particles,
        'model': args.model_list,
        'lambda_diag': args.lambda_diag,
        'lambda_off': args.lambda_off,
        'sigma_beta': 1.0,
        'ess_threshold_factor': 0.5,
        'mcmc_step_size': 0.5,
        'dataset_path': args.data,
        'update_human_params': args.update_human,
        'update_llm_params': args.update_llm,
        'use_swap_logits_ba': args.use_swap_logits_ba,
        'zero_human_likelihood': args.zero_human_likelihood,
        'zero_llm_likelihood': args.zero_llm_likelihood,
        'human_priors': {k: v.tolist() for k, v in (human_priors or {}).items()},
        'human_judges': args.human_judges_list,
    }
    
    print("=" * 70)
    print("SummEval Experiment - Train/Test Split")
    print("=" * 70)
    
    if CONFIG['zero_human_likelihood']:
        print("[Note] Human likelihood set to 0 in test phase (ablation mode)")
        print()
    if CONFIG['zero_llm_likelihood']:
        print("[Note] LLM likelihood set to 0 in test phase (ablation mode)")
        print()
    
    print_experiment_config(CONFIG)

    train_samples = samples[:args.n_train]
    test_samples = samples[args.n_train:]
    
    print(f"[Data Split]")
    print(f"  Total Samples: {n_total}")
    print(f"  Train Samples: {len(train_samples)} (update framework params)")
    print(f"  Test Samples: {len(test_samples)} (fixed params, inference only)")
    print()
    
    evaluator = StrictBayesianEvaluator(
        n_particles=CONFIG['n_particles'],
        lambda_diag=CONFIG['lambda_diag'],
        lambda_off=CONFIG['lambda_off'],
        sigma_beta=CONFIG['sigma_beta'],
        ess_threshold_factor=CONFIG['ess_threshold_factor'],
        mcmc_step_size=CONFIG['mcmc_step_size'],
        verbose=CONFIG['verbose'],
        max_print_raters=CONFIG['max_print_raters'],
        print_interval=CONFIG['print_interval'],
        update_human_params=CONFIG['update_human_params'],
        update_llm_params=CONFIG['update_llm_params'],
        use_swap_logits_ba=CONFIG['use_swap_logits_ba'],
        zero_llm_likelihood=CONFIG['zero_llm_likelihood'],
        human_priors=human_priors
    )
    
    visualizer = None
    beta_visualizer = None
    if CONFIG["live_plot"]:
        try:
            visualizer = VarianceConvergencePlotter(max_samples=n_total)
            beta_visualizer = BetaConvergencePlotter(max_samples=n_total)
        except Exception as e:
            print(f"[Warning] Visualizer initialization failed: {e}")
    
    sample_results = []
    
    print("=" * 70)
    print("[Training Phase] Update framework parameters")
    print("=" * 70)
    
    live_llm_model = args.live_llm_model
    if args.live_llm and live_llm_model is None:
        live_llm_model = args.model_list[0] if args.model_list else "gpt-4o-mini"

    for idx, sample in enumerate(train_samples):
        current_sample_num = idx + 1

        if args.live_llm:
            label_ab, label_ba = get_live_llm_judgment(sample, live_llm_model)
            sample.llm_judgments[live_llm_model] = (label_ab, label_ba)

        evaluator.update_online(sample)
        
        sample_verdict = evaluator.get_sample_verdict()
        
        llm_beta_stats = {}
        for llm_id in evaluator.beta_particles.keys():
            stats = evaluator.get_llm_posterior_stats(llm_id)
            if stats:
                llm_beta_stats[llm_id] = {
                    'mean': stats['mean'],
                    'variance': stats['variance']
                }
        
        learned_C_params = {}
        for rater_id, alpha_mat in evaluator.C_params.items():
            learned_C_params[rater_id] = (alpha_mat / alpha_mat.sum(axis=1, keepdims=True)).tolist()
        
        current_ess = {}
        for llm_id in evaluator.ess_history.keys():
            if evaluator.ess_history[llm_id]:
                current_ess[llm_id] = float(evaluator.ess_history[llm_id][-1])
        
        sample_result = {
            'sample_idx': current_sample_num,
            'phase': 'train',
            'metadata': sample.metadata,
            'human_judgments': sample.human_judgments,
            'llm_judgments': {
                llm_id: {'label_AB': label[0], 'label_BA': label[1]}
                for llm_id, label in sample.llm_judgments.items()
            },
            'verdict': {
                'pi_est': sample_verdict['pi_est'],
                'eta': sample_verdict.get('eta', evaluator._current_eta.tolist() if evaluator._current_eta is not None else [0, 0]),
                'winner': sample_verdict['winner'],
                'confidence': sample_verdict['confidence'],
                'entropy': sample_verdict['entropy'],
                'decision_advice': sample_verdict['decision_advice'],
            },
            'llm_beta_stats': llm_beta_stats,
            'learned_C_params': learned_C_params,
            'ess': current_ess
        }
        sample_results.append(sample_result)
        
        if visualizer:
            visualizer.update(current_sample_num, evaluator)
        if beta_visualizer:
            beta_visualizer.update(current_sample_num, evaluator)
    
    print(f"\nTraining complete! Processed {len(train_samples)} samples")
    
    if CONFIG['zero_human_likelihood'] and CONFIG['zero_llm_likelihood']:
        evaluator.zero_human_likelihood = True
        evaluator.zero_llm_likelihood = True
        print("\n" + "=" * 70)
        print("[Test Phase] Fixed parameters, inference only (human+LLM likelihood set to 0)")
        print("=" * 70)
    elif CONFIG['zero_human_likelihood']:
        evaluator.zero_human_likelihood = True
        print("\n" + "=" * 70)
        print("[Test Phase] Fixed parameters, inference only (human likelihood set to 0)")
        print("=" * 70)
    elif CONFIG['zero_llm_likelihood']:
        evaluator.zero_llm_likelihood = True
        print("\n" + "=" * 70)
        print("[Test Phase] Fixed parameters, inference only (LLM likelihood set to 0)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("[Test Phase] Fixed parameters, inference only")
        print("=" * 70)
    
    test_total_log_lik = 0
    test_human_total_log_lik = 0
    test_llm_total_log_lik = 0

    test_log_lik_list = []
    test_human_log_lik_list = []
    test_llm_log_lik_list = []
    for idx, sample in enumerate(test_samples):
        current_sample_num = len(train_samples) + idx + 1
        
        sample_verdict = evaluator.infer_with_fixed_params(
            human_judgments=sample.human_judgments,
            llm_judgments=sample.llm_judgments
        )
        
        llm_beta_stats = {}
        for llm_id in evaluator.beta_particles.keys():
            stats = evaluator.get_llm_posterior_stats(llm_id)
            if stats:
                llm_beta_stats[llm_id] = {
                    'mean': stats['mean'],
                    'variance': stats['variance']
                }
        
        learned_C_params = {}
        for rater_id, alpha_mat in evaluator.C_params.items():
            learned_C_params[rater_id] = (alpha_mat / alpha_mat.sum(axis=1, keepdims=True)).tolist()
        
        current_ess = {}
        for llm_id in evaluator.ess_history.keys():
            if evaluator.ess_history[llm_id]:
                current_ess[llm_id] = float(evaluator.ess_history[llm_id][-1])
        
        loho_predictions = {}
        human_rater_ids = list(sample.human_judgments.keys())
        for left_out_rater in human_rater_ids:
            human_judgments_loho = {
                k: v for k, v in sample.human_judgments.items()
                if k != left_out_rater
            }
            verdict_loho = evaluator.infer_with_fixed_params(
                human_judgments=human_judgments_loho,
                llm_judgments=sample.llm_judgments
            )
            pi_loho = np.array(verdict_loho['pi_est'])
            if left_out_rater in evaluator.C_params:
                alpha_mat = evaluator.C_params[left_out_rater]
                C_r = alpha_mat / alpha_mat.sum(axis=1, keepdims=True)
            else:
                C_r = evaluator._get_prior_C_estimate(left_out_rater)
            mu_r = C_r.T @ pi_loho
            pred_idx = int(np.argmax(mu_r))
            pred_label = LabelMapping.from_index(pred_idx)
            true_label = sample.human_judgments[left_out_rater]
            true_idx = LabelMapping.to_index(true_label)
            loho_predictions[left_out_rater] = {
                'true': true_label,
                'predicted': pred_label,
                'correct': pred_idx == true_idx,
                'mu': mu_r.tolist(),
                'pi_loho': pi_loho.tolist()
            }

        sample_log_lik = sample_verdict.get('predictive_log_likelihood', None)
        sample_human_log_lik = sample_verdict.get('human_log_likelihood', None)
        sample_llm_log_lik = sample_verdict.get('llm_log_likelihood', None)

        if sample_log_lik is not None:
            test_total_log_lik += sample_log_lik
            test_log_lik_list.append(sample_log_lik)
        
        if sample_human_log_lik is not None:
            test_human_total_log_lik += sample_human_log_lik
            test_human_log_lik_list.append(sample_human_log_lik)

        if sample_llm_log_lik is not None:
            test_llm_total_log_lik += sample_llm_log_lik
            test_llm_log_lik_list.append(sample_llm_log_lik)
        
        sample_result = {
            'sample_idx': current_sample_num,
            'phase': 'test',
            'metadata': sample.metadata,
            'human_judgments': sample.human_judgments,
            'llm_judgments': {
                llm_id: {'label_AB': label[0], 'label_BA': label[1]}
                for llm_id, label in sample.llm_judgments.items()
            },
            'verdict': {
                'pi_est': sample_verdict['pi_est'],
                'eta': sample_verdict.get('eta', [0, 0]),
                'winner': sample_verdict['winner'],
                'confidence': sample_verdict['confidence'],
                'entropy': sample_verdict['entropy'],
                'decision_advice': sample_verdict['decision_advice'],
                'predictive_log_likelihood': sample_log_lik
            },
            'llm_beta_stats': llm_beta_stats,
            'learned_C_params': learned_C_params,
            'ess': current_ess,
            'loho_predictions': loho_predictions
        }
        sample_results.append(sample_result)
        
        if visualizer:
            visualizer.update(current_sample_num, evaluator)
        if beta_visualizer:
            beta_visualizer.update(current_sample_num, evaluator)
    
    print(f"\nTest complete! Processed {len(test_samples)} samples")
    
    loho_stats = {}
    loho_correct_by_rater = {}
    loho_total_by_rater = {}
    loho_correct_overall = 0
    loho_total_overall = 0
    for sr in sample_results:
        if sr.get('phase') != 'test':
            continue
        loho_preds = sr.get('loho_predictions', {})
        for rater, pred_info in loho_preds.items():
            loho_total_by_rater[rater] = loho_total_by_rater.get(rater, 0) + 1
            loho_total_overall += 1
            if pred_info.get('correct', False):
                loho_correct_by_rater[rater] = loho_correct_by_rater.get(rater, 0) + 1
                loho_correct_overall += 1
    
    loho_overall_acc = loho_correct_overall / loho_total_overall if loho_total_overall > 0 else 0.0
    loho_stats['overall_accuracy'] = float(loho_overall_acc)
    loho_stats['per_rater'] = {}
    for rater in loho_total_by_rater.keys():
        acc = loho_correct_by_rater.get(rater, 0) / loho_total_by_rater[rater]
        loho_stats['per_rater'][rater] = {
            'correct': loho_correct_by_rater.get(rater, 0),
            'total': loho_total_by_rater[rater],
            'accuracy': float(acc)
        }
    
    if len(test_log_lik_list) > 0:
        test_avg_log_lik = test_total_log_lik / len(test_log_lik_list)
        test_std_log_lik = np.std(test_log_lik_list)
        test_perplexity = np.exp(-test_avg_log_lik)
    else:
        test_avg_log_lik = 0.0
        test_std_log_lik = 0.0
        test_perplexity = 1.0
    
    print("\n" + "-" * 70)
    print("[Test Set Predictive Log-Likelihood Statistics]")
    print(f"  Total Log-Likelihood: {test_total_log_lik:.4f}")
    print(f"  Mean Log-Likelihood: {test_avg_log_lik:.4f}")
    print(f"  Std Dev: {test_std_log_lik:.4f}")
    print(f"  Perplexity: {test_perplexity:.4f}")
    print("-" * 70)
    
    print("\n" + "-" * 70)
    print("[LOHO (Leave-One-Human-Out) Prediction Accuracy]")
    print(f"  Overall Accuracy: {loho_overall_acc:.1%} ({loho_correct_overall}/{loho_total_overall})")
    for rater, info in loho_stats['per_rater'].items():
        print(f"  {rater:<12} Accuracy: {info['accuracy']:.1%} ({info['correct']}/{info['total']})")
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Evaluation complete, generating report...")
    print("=" * 70)
    
    report = evaluator.generate_learning_report()
    print("\n" + report)
    
    llm_stats = {}
    for llm_id in evaluator.beta_particles.keys():
        llm_stats[llm_id] = evaluator.get_llm_posterior_stats(llm_id)
    
    beta_history_data = {}
    for llm_id in evaluator.beta_history.keys():
        beta_history_data[llm_id] = {
            'beta_mean_history': evaluator.beta_history[llm_id],
            'variance_history': evaluator.variance_history.get(llm_id, []),
            'ess_history': evaluator.ess_history.get(llm_id, [])
        }
    
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    uh = "uh1" if CONFIG['update_human_params'] else "uh0"
    ul = "ul1" if CONFIG['update_llm_params'] else "ul0"
    
    if args.output_name is not None:
        file_prefix = args.output_name
    else:
        file_prefix = f"{CONFIG['experiment_name']}_{uh}_{ul}"
    
    if visualizer:
        visualizer.finalize(os.path.join(output_dir, f"{file_prefix}_variance.png"))
    if beta_visualizer:
        beta_visualizer.finalize(os.path.join(output_dir, f"{file_prefix}_beta.png"))
    
    final_C_params = {}
    for rater_id, alpha_mat in evaluator.C_params.items():
        final_C_params[rater_id] = (alpha_mat / alpha_mat.sum(axis=1, keepdims=True)).tolist()
    
    final_llm_beta_summary = {}
    for llm_id in evaluator.beta_particles.keys():
        stats = evaluator.get_llm_posterior_stats(llm_id)
        if stats:
            final_llm_beta_summary[llm_id] = {
                'mean': stats['mean'],
                'variance': stats['variance']
            }
    
    
    results = {
        "experiment_info": {
            "name": CONFIG['experiment_name'],
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        },
        "config": CONFIG,
        "summary": {
            "model_win_rate": evaluator.get_model_win_rate(),
            "total_samples": n_total,
            "train_samples": args.n_train,
            "test_samples": n_test,
            "final_llm_beta": final_llm_beta_summary,
            "final_C_params": final_C_params,
            "test_predictive_log_likelihood": {
                "total": float(test_total_log_lik),
                "human": float(test_human_total_log_lik),
                "llm": float(test_llm_total_log_lik),
                "mean": float(test_avg_log_lik),
                "std": float(test_std_log_lik),
                "perplexity": float(test_perplexity),
                "n_samples": len(test_log_lik_list)
            },
            "loho_accuracy": loho_stats
        },
        "sample_results": sample_results,
        "beta_history": beta_history_data,
        "eta_history": evaluator.pi_history if hasattr(evaluator, 'pi_history') else []
    }
    
    if args.output_name is not None:
        file_prefix = args.output_name
    else:
        file_prefix = f"{CONFIG['experiment_name']}_{uh}_{ul}"
    
    results_path = os.path.join(output_dir, f"{file_prefix}_results.json")
    
    print(f"[Output Directory] {output_dir}")
    results_native = convert_to_native(results)
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_native, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("[Experiment Result Summary]")
    print("=" * 70)
    win_rates = results['summary']['model_win_rate']
    print(f"  Total Samples:         {n_total}")
    print(f"  Train Samples:         {args.n_train}")
    print(f"  Test Samples:          {n_test}")
    print(f"  Model A Win Rate:      {win_rates.get('win_rate_A', 0):.3f}")
    print(f"  Model B Win Rate:      {win_rates.get('win_rate_B', 0):.3f}")
    print(f"  Tie Rate:              {win_rates.get('tie_rate', 0):.3f}")
    print()
    print(f"  Test Total Log-Lik:    {test_total_log_lik:.4f}")
    print(f"  Test Mean Log-Lik:     {test_avg_log_lik:.4f}")
    if len(test_human_log_lik_list) > 0:
        print(f"  Test Mean Human Log-Lik: {test_human_total_log_lik / len(test_human_log_lik_list):.4f}")
    if len(test_llm_log_lik_list) > 0:
        print(f"  Test Mean LLM Log-Lik:   {test_llm_total_log_lik / len(test_llm_log_lik_list):.4f}")
    print(f"  Test Perplexity:       {test_perplexity:.4f}")
    print(f"  Result File:           {results_path}")
    print("=" * 70)    
    
    print("\nAll processing complete!")
    
    if hasattr(sys.stdout, 'close') and isinstance(sys.stdout, Tee):
        sys.stdout.close()
        sys.stdout = sys.stdout.console


if __name__ == "__main__":
    main()
