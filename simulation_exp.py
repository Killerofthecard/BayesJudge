#!/usr/bin/env python3
"""
Simulation Experiment with Train/Test Split
First 400 samples for training (parameter updates), last 100 samples for testing (fixed parameters, inference only)
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime
from pathlib import Path

from bayesian_eval.data.data_types import SampleData
from bayesian_eval.core.evaluator import StrictBayesianEvaluator

try:
    from bayesian_eval.visualization.variance_plotter import VarianceConvergencePlotter
    from bayesian_eval.visualization.beta_plotter import BetaConvergencePlotter
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


def load_synthetic_dataset(json_path: str, llm_models: List[str] = None) -> List[SampleData]:
    """Load synthetic dataset from a JSON file, with optional filtering by specified LLMs."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    dataset = []
    for item in data_list:
        metadata = item.get('metadata')
        conversation_a = item.get('conversation_a')
        conversation_b = item.get('conversation_b')
        
        llm_judgments_raw = item.get('llm_judgments', {})
        llm_judgments = {}
        for llm_id, judgment in llm_judgments_raw.items():
            if llm_models is not None and llm_id not in llm_models:
                continue
            llm_judgments[llm_id] = (judgment[0], judgment[1])
        
        sample = SampleData(
            human_judgments=item.get('human_judgments', {}),
            llm_judgments=llm_judgments,
            metadata=metadata,
            conversation_a=conversation_a,
            conversation_b=conversation_b
        )
        dataset.append(sample)
    
    print(f"[Data Loading] Successfully loaded {len(dataset)} samples from {json_path}")
    if llm_models is not None:
        print(f"[Data Loading] Specified LLMs for modeling: {', '.join(llm_models)}")
    return dataset


def compute_frobenius_norm_diff(matrix_learned: np.ndarray, matrix_preset: np.ndarray) -> float:
    """Compute the Frobenius norm difference between two matrices."""
    diff = matrix_learned - matrix_preset
    return np.linalg.norm(diff, 'fro')


def load_preset_confusion_matrices(config_path: str) -> dict:
    """Load preset human confusion matrices from a config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    human_config = config.get('human_rater_config', {})
    preset_matrices = {}
    
    raters_list = human_config.get('raters', [])
    for rater_config in raters_list:
        rater_name = rater_config.get('name')
        confusion_matrix = rater_config.get('confusion_matrix', None)
        if rater_name is not None and confusion_matrix is not None:
            preset_matrices[rater_name] = np.array(confusion_matrix)
    
    for old_rater_id in ['rater_0', 'rater_1']:
        if old_rater_id in human_config:
            rater_config = human_config[old_rater_id]
            confusion_matrix = rater_config.get('confusion_matrix', None)
            if confusion_matrix is not None:
                preset_matrices[old_rater_id] = np.array(confusion_matrix)
    
    return preset_matrices


def load_preset_beta_values(config_path: str) -> dict:
    """Load preset LLM beta parameters from a config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    llm_config = config.get('llm_config', {})
    preset_betas = {}
    
    llms_list = llm_config.get('llms', [])
    for llm_config_item in llms_list:
        llm_name = llm_config_item.get('name')
        beta = llm_config_item.get('beta', None)
        if llm_name is not None and beta is not None:
            preset_betas[llm_name] = float(beta)
    
    return preset_betas


def convert_to_native(obj):
    """Recursively convert numpy types to Python native types."""
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


def print_experiment_config(config: dict):
    """Pretty-print experiment parameters."""
    print()
    print("[Experiment Configuration]")
    print("-" * 70)
    print(f"  Experiment Name:  {config['experiment_name']}")
    print(f"  Data File:        {config['dataset_path']}")
    print(f"  Evaluation Model: {config['model']}")
    print()
    print("  [Sampling Configuration]")
    print(f"    Total Samples:  {config['n_samples']}")
    print(f"    Train Samples:  400")
    print(f"    Test Samples:   {config['n_samples'] - 400}")
    print()
    print("  [Particle Filter Configuration]")
    print(f"    Particles:      {config['n_particles']}")
    print(f"    Lambda(Diag):   {config['lambda_diag']}")
    print(f"    Lambda(Off):    {config['lambda_off']}")
    print(f"    Sigma Beta:     {config['sigma_beta']}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Simulation Experiment with Train/Test Split")
    parser.add_argument("--data", type=str, default="./dataset/synthetic_1000_custom.json",
                       help="Path to the data file")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to the config file")
    parser.add_argument("--n_train", type=int, default=800,
                       help="Number of training samples (default: 400)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="LLM model for evaluation (kept for compatibility, not used for synthetic data filtering)")
    parser.add_argument("--llm_models", type=str, default=None,
                       help="Comma-separated list of simulated LLMs for joint modeling (e.g., llm_strong,llm_weak). Default: use all")
    parser.add_argument("--n_particles", type=int, default=300,
                       help="Number of particles")
    parser.add_argument("--lambda_diag", type=float, default=1.0,
                       help="Diagonal element initialization for human confusion matrix (default: 1.0)")
    parser.add_argument("--lambda_off", type=float, default=0.0,
                       help="Off-diagonal element initialization for human confusion matrix (default: 0.0)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: framework built-in outputs/ directory)")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Output filename prefix (default: framework built-in naming rule)")
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
                       help="Use swapped logits_ba for LLM likelihood computation (default: True)")
    parser.add_argument("--no_swap_logits_ba", action="store_false", dest="use_swap_logits_ba",
                       help="Disable swapped logits_ba, use non-swapped version")
    
    args = parser.parse_args()
    
    args.llm_models_list = None
    if args.llm_models is not None:
        args.llm_models_list = [m.strip() for m in args.llm_models.split(',') if m.strip()]
    
    def resolve_path(p):
        if p is None:
            return None
        path = Path(p)
        if not path.is_absolute():
            path = Path(PROJECT_ROOT) / path
        return path.resolve()
    
    args.data = str(resolve_path(args.data))
    args.config = str(resolve_path(args.config)) if args.config else None
    args.output_dir = str(resolve_path(args.output_dir)) if args.output_dir else None
    
    if args.config is None:
        data_path = Path(args.data)
        config_path = data_path.parent / f"{data_path.stem}_config.json"
    else:
        config_path = Path(args.config)
    
    preset_matrices = {}
    preset_betas = {}
    if config_path.exists():
        preset_matrices = load_preset_confusion_matrices(str(config_path))
        preset_betas = load_preset_beta_values(str(config_path))
        print(f"[Config] Loaded preset config: {config_path}")
        print(f"  - Human confusion matrices: {len(preset_matrices)} raters")
        print(f"  - LLM beta parameters: {preset_betas}")
    
    samples = load_synthetic_dataset(args.data, llm_models=args.llm_models_list)
    n_total = len(samples)
    n_train = min(args.n_train, n_total)
    n_test = n_total - n_train
    
    CONFIG = {
        'experiment_name': 'simulation_train_test',
        'verbose': not args.no_verbose,
        'print_interval': 1,
        'max_print_raters': 2,
        'live_plot': args.live_plot and HAS_VISUALIZATION,
        'n_samples': n_total,
        'n_train': n_train,
        'n_test': n_test,
        'n_particles': args.n_particles,
        'model': args.model,
        'llm_models': args.llm_models_list,
        'lambda_diag': args.lambda_diag,
        'lambda_off': args.lambda_off,
        'sigma_beta': 1.0,
        'ess_threshold_factor': 0.5,
        'mcmc_step_size': 0.5,
        'dataset_path': args.data,
        'config_path': str(config_path) if config_path.exists() else None,
        'update_human_params': args.update_human,
        'update_llm_params': args.update_llm,
        'use_swap_logits_ba': args.use_swap_logits_ba,
    }
    
    print("=" * 70)
    print("Simulation Experiment - Train/Test Split")
    print("=" * 70)
    
    print_experiment_config(CONFIG)
    
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]
    
    print(f"[Data Split]")
    print(f"  Total Samples:  {n_total}")
    print(f"  Train Samples:  {len(train_samples)} (update framework parameters)")
    print(f"  Test Samples:   {len(test_samples)} (fixed parameters, inference only)")
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
        use_swap_logits_ba=CONFIG['use_swap_logits_ba']
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
    print("[Training Phase] Updating framework parameters")
    print("=" * 70)
    
    for idx, sample in enumerate(train_samples):
        current_sample_num = idx + 1
        
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
            'ess': current_ess
        }
        sample_results.append(sample_result)
        
        if visualizer:
            visualizer.update(current_sample_num, evaluator)
        if beta_visualizer:
            beta_visualizer.update(current_sample_num, evaluator)
    
    print(f"\nTesting complete! Processed {len(test_samples)} samples")
    
    if preset_betas:
        print("\n" + "=" * 70)
        print("[LLM Beta Parameter Comparison]")
        print("=" * 70)
        print(f"{'LLM ID':<20} {'Preset Beta':>12} {'Learned Mean':>12} {'Abs Error':>12} {'Rel Error':>12}")
        print("-" * 70)
        
        beta_errors = []
        for llm_id, preset_beta in preset_betas.items():
            if llm_id in evaluator.beta_particles:
                particles = evaluator.beta_particles[llm_id]
                weights = evaluator.weights[llm_id]
                learned_mean = np.average(particles, weights=weights)
                
                abs_error = abs(learned_mean - preset_beta)
                rel_error = abs_error / abs(preset_beta) if preset_beta != 0 else float('inf')
                
                beta_errors.append((preset_beta - learned_mean) ** 2)
                
                print(f"{llm_id:<20} {preset_beta:>12.4f} {learned_mean:>12.4f} {abs_error:>12.4f} {rel_error:>12.2%}")
            else:
                print(f"{llm_id:<20} {preset_beta:>12.4f} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
        
        if beta_errors:
            mse = np.mean(beta_errors)
            rmse = np.sqrt(mse)
            print("-" * 70)
            print(f"  MSE:  {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
        print("=" * 70)
    
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
    print(f"  Mean Log-Likelihood:  {test_avg_log_lik:.4f}")
    print(f"  Std Dev:              {test_std_log_lik:.4f}")
    print(f"  Perplexity:           {test_perplexity:.4f}")
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
    
    ground_truth = {}
    if preset_matrices:
        ground_truth['human_C_params'] = {
            rater_id: matrix.tolist() 
            for rater_id, matrix in preset_matrices.items()
        }
    if preset_betas:
        ground_truth['llm_beta'] = preset_betas
    
    results = {
        "experiment_info": {
            "name": CONFIG['experiment_name'],
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        },
        "config": CONFIG,
        "ground_truth": ground_truth,
        "summary": {
            "model_win_rate": evaluator.get_model_win_rate(),
            "total_samples": n_total,
            "train_samples": n_train,
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
            }
        },
        "sample_results": sample_results,
        "beta_history": beta_history_data,
        "eta_history": evaluator.pi_history if hasattr(evaluator, 'pi_history') else []
    }
    
    uh = "uh1" if CONFIG['update_human_params'] else "uh0"
    ul = "ul1" if CONFIG['update_llm_params'] else "ul0"
    
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
    print(f"  Total Samples:           {n_total}")
    print(f"  Train Samples:           {n_train}")
    print(f"  Test Samples:            {n_test}")
    print(f"  Model A Win Rate:        {win_rates.get('win_rate_A', 0):.3f}")
    print(f"  Model B Win Rate:        {win_rates.get('win_rate_B', 0):.3f}")
    print(f"  Tie Rate:                {win_rates.get('tie_rate', 0):.3f}")
    print()
    print(f"  Test Total Log-Likelihood:     {test_total_log_lik:.4f}")
    print(f"  Test Mean Log-Likelihood:      {test_avg_log_lik:.4f}")
    print(f"  Test Mean Human Log-Likelihood: {test_human_total_log_lik / len(test_human_log_lik_list):.4f}")
    print(f"  Test Mean LLM Log-Likelihood:   {test_llm_total_log_lik / len(test_llm_log_lik_list):.4f}")
    print(f"  Test Perplexity:               {test_perplexity:.4f}")
    print(f"  Result File:                   {results_path}")
    print("=" * 70)
    
    if preset_matrices:
        print("\n" + "=" * 70)
        print("[Human Confusion Matrix Comparison]")
        print("=" * 70)
        
        for rater_id, preset_matrix in preset_matrices.items():
            if rater_id in evaluator.C_params:
                alpha_mat = evaluator.C_params[rater_id]
                learned_matrix = alpha_mat / alpha_mat.sum(axis=1, keepdims=True)
                f_norm = compute_frobenius_norm_diff(learned_matrix, preset_matrix)
                
                print(f"\n{rater_id}:")
                print(f"  Frobenius Norm: {f_norm:.6f}")
                
                print(f"  Learned Matrix:")
                for i, row in enumerate(learned_matrix):
                    print(f"    {['A', 'C', 'B'][i]}: " + " | ".join([f"{x:.3f}" for x in row]))
                
                print(f"  Preset Matrix:")
                for i, row in enumerate(preset_matrix):
                    print(f"    {['A', 'C', 'B'][i]}: " + " | ".join([f"{x:.3f}" for x in row]))
        
        print("=" * 70)
    
    print("\nAll processing complete!")


if __name__ == "__main__":
    main()
