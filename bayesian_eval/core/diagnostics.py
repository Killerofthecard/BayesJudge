"""
Diagnostics and report generation.
Corresponds to Section 6 of the paper: final evaluation results and diagnostics.
"""

import numpy as np
from typing import Dict, List, Tuple
from ..data.data_types import SampleData, LabelMapping
from .likelihood import softmax_transform


class DiagnosticsPrinter:
    """Diagnostic information printer."""
    
    def __init__(self, max_print_raters: int = 2, print_interval: int = 1):
        self.max_print_raters = max_print_raters
        self.print_interval = print_interval
        self.sep = "=" * 80
    
    def print_sample_report(
        self,
        t: int,
        sample: SampleData,
        C_params: Dict[str, np.ndarray],
        beta_particles: Dict[str, np.ndarray],
        weights: Dict[str, np.ndarray],
        current_eta: np.ndarray,
        M: int,
        get_llm_stats_fn
    ):
        """Print a complete evaluation report for a single sample."""
        if t % self.print_interval != 0:
            return

        human_judgments = sample.human_judgments
        llm_judgments = sample.llm_judgments

        print(f"\n{self.sep}")
        q_id = sample.get_question_id()
        print(f"[Evaluation Sample #{t}] Question ID: {q_id}")
        print(f"{self.sep}\n")

        self._print_swap_design(llm_judgments)
        self._print_verdict(current_eta)
        self._print_human_judgments(human_judgments, C_params)
        self._print_c_matrices(C_params)
        self._print_llm_bias(beta_particles, weights, M, get_llm_stats_fn)

        print(f"\n{self.sep}\n")
    
    def _print_swap_design(self, llm_judgments: Dict[str, Tuple[str, str]]):
        """Print swap-design evaluation results."""
        print("▸ Original Evaluation Results (Swap Design)")
        print("-" * 80)
        print("  Option meaning: [A]=Choose position 1 (first answer shown)")
        print("                  [B]=Choose position 2 (second answer shown)")
        print("  Sample mapping: AB order -> position1=A,position2=B | BA order -> position1=B,position2=A")
        print()

        if llm_judgments:
            print("  LLM Pairwise Swap Evaluation:")
            for llm_id, (label_ab, label_ba) in llm_judgments.items():
                sample_selected_ab = label_ab
                sample_selected_ba = "B" if label_ba == "A" else "A"

                if sample_selected_ab == sample_selected_ba:
                    print(f"    ✓ {llm_id}  [No Position Bias]")
                    print(f"       ├─ AB order: option '{label_ab}' -> sample {sample_selected_ab}")
                    print(f"       ├─ BA order: option '{label_ba}' -> sample {sample_selected_ba}")
                else:
                    if label_ab == "A" and label_ba == "A":
                        pos = "Position 1 (shown first)"
                        beta_sign = "β > 0"
                    elif label_ab == "B" and label_ba == "B":
                        pos = "Position 2 (shown second)"
                        beta_sign = "β < 0"
                    else:
                        pos = "Unknown position"
                        beta_sign = "?"

                    print(f"    ⚠️ {llm_id}  [Position Bias Conflict]")
                    print(f"       ├─ AB order: option '{label_ab}' -> sample {sample_selected_ab}")
                    print(f"       ├─ BA order: option '{label_ba}' -> sample {sample_selected_ba}")
                    print(f"       └─ Interpretation: Always chooses [{pos}] -> {beta_sign}")
        print()
    
    def _print_verdict(self, current_eta: np.ndarray):
        """Print the final verdict distribution."""
        print("▸ Final Verdict Distribution pi_t (Bayesian Posterior Estimate)")
        print("-" * 80)

        pi = softmax_transform(current_eta)
        labels = ["A better than B", "Tie", "B better than A"]

        for i, (label, prob) in enumerate(zip(labels, pi)):
            bar_len = int(prob * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            marker = "→" if i == np.argmax(pi) else " "
            symbol = "A" if i == 0 else ("T" if i == 1 else "B")
            print(f"  {marker} {label:<12} [{symbol}] {bar}  {prob:>6.2%}")

        winner_idx = int(np.argmax(pi))
        winner = LabelMapping.from_index(winner_idx)
        confidence = float(pi[winner_idx])
        
        print(f"\n  Decision Advice: {'High-confidence verdict' if confidence > 0.6 else 'Need more evidence'}")
        entropy = -np.sum(pi * np.log(pi + 1e-10))
        print(f"  Distribution Entropy: {entropy:.4f}")
        print()
    
    def _print_human_judgments(
        self,
        human_judgments: Dict[str, str],
        C_params: Dict[str, np.ndarray]
    ):
        """Print human evaluation results."""
        print("▸ Human Expert Evaluation")
        print("-" * 80)
        if human_judgments:
            for rater_id, label in sorted(human_judgments.items()):
                if rater_id in C_params:
                    alpha_mat = C_params[rater_id]
                    C_mat = alpha_mat / alpha_mat.sum(axis=1, keepdims=True)
                    accuracy = np.mean([C_mat[i, i] for i in range(3)])
                    status = f"[Learned, estimated accuracy={accuracy:.0%}]"
                else:
                    status = "[Prior initialization]"
                print(f"    • {rater_id:<12} -> Verdict: {label} {status}")
        else:
            print("    (No human annotations for this sample)")
        print()
    
    def _print_c_matrices(self, C_params: Dict[str, np.ndarray]):
        """Print human confusion matrices."""
        print("▸ Cumulative Learning: Human Rater Confusion Matrix C_r (Posterior Mean)")
        print("-" * 80)

        if C_params:
            for rater_id in sorted(C_params.keys())[:self.max_print_raters]:
                alpha_mat = C_params[rater_id]
                C_mat = alpha_mat / alpha_mat.sum(axis=1, keepdims=True)

                accuracy = np.mean([C_mat[i, i] for i in range(3)])
                lambda_diag = 2.0
                lambda_off = 1.0
                total_updates = int(alpha_mat.sum() - 3 * (lambda_diag + 2 * lambda_off))

                print(f"\n  Rater: {rater_id} (Learned samples: {total_updates}, Average accuracy: {accuracy:.1%})")
                print(f"                {'Observed verdict ->':<45}")
                print(f"                {'A':>10} {'Tie':>10} {'B':>10}  {'(True quality)':<12}")
                print(f"  {'True A better':<12}  [{C_mat[0, 0]:>8.3f}  {C_mat[0, 1]:>8.3f}  {C_mat[0, 2]:>8.3f}]")
                print(f"  {'True Tie':<12}  [{C_mat[1, 0]:>8.3f}  {C_mat[1, 1]:>8.3f}  {C_mat[1, 2]:>8.3f}]")
                print(f"  {'True B better':<12}  [{C_mat[2, 0]:>8.3f}  {C_mat[2, 1]:>8.3f}  {C_mat[2, 2]:>8.3f}]")
        else:
            print("  (No cumulative data yet)")
        print()
    
    def _print_llm_bias(
        self,
        beta_particles: Dict[str, np.ndarray],
        weights: Dict[str, np.ndarray],
        M: int,
        get_llm_stats_fn
    ):
        """Print LLM position bias."""
        print("▸ Cumulative Learning: LLM Position Bias beta")
        print("-" * 80)

        if beta_particles:
            for llm_id in sorted(beta_particles.keys()):
                stats = get_llm_stats_fn(llm_id)
                if not stats:
                    continue

                status_icon = "✓" if stats['converged'] else "○"
                status_text = "Converged" if stats['converged'] else "Learning"

                beta_mean = stats['mean']
                if abs(beta_mean) < 0.1:
                    bias_desc = "No significant position bias (stable judgment)"
                elif beta_mean > 0:
                    bias_desc = f"Prefers [first-shown position] (β=+{beta_mean:.2f})"
                else:
                    bias_desc = f"Prefers [second-shown position] (β={beta_mean:.2f})"

                print(f"  {status_icon} {llm_id:<15}")
                print(f"    ├─ beta estimate: {beta_mean:>+.3f} +- {stats['std']:.3f} [{status_text}]")
                print(f"    ├─ 95% credible interval: [{stats['ci_95'][0]:>+.3f}, {stats['ci_95'][1]:>+.3f}]")
                print(f"    ├─ Interpretation: {bias_desc}")
                ess_status = "Good" if stats['ess'] > M * 0.5 else "Needs resampling"
                print(f"    └─ Particle health: ESS={stats['ess']:>6.1f}/{M} ({ess_status})")
        else:
            print("  (No LLM evaluation data yet)")


def generate_learning_report(
    beta_particles: Dict[str, np.ndarray],
    weights: Dict[str, np.ndarray],
    beta_history: Dict[str, List],
    pi_history: List[np.ndarray],
    M: int,
    get_llm_stats_fn
) -> str:
    """    
    Returns:
        Report text.
    """
    report_lines = ["=== Strict Bayesian Evaluation Cumulative Learning Report ===\n"]

    for llm_id in beta_particles.keys():
        stats = get_llm_stats_fn(llm_id)
        report_lines.append(f"LLM Judge: {llm_id}")
        report_lines.append(f"  Final position bias estimate: {stats['mean']:.4f} ± {stats['std']:.4f}")
        report_lines.append(f"  95% credible interval: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")
        report_lines.append(f"  Convergence status: {'Converged' if stats['converged'] else 'Not converged'}")
        report_lines.append(f"  Final ESS: {stats['ess']:.1f} / {M}")

        if len(beta_history.get(llm_id, [])) >= 20:
            early_var = np.var(beta_history[llm_id][:10])
            late_var = np.var(beta_history[llm_id][-10:])
            report_lines.append(f"  Variance decay ratio (first 10 / last 10): {early_var / max(late_var, 1e-10):.2f}x")
        report_lines.append("")

    if pi_history:
        T = len(pi_history)
        pi_array = np.array(pi_history)
        win_rate_A = np.mean(pi_array[:, 0])
        win_rate_B = np.mean(pi_array[:, 2])
        tie_rate = np.mean(pi_array[:, 1])

        report_lines.append("Model-level expected win rates:")
        report_lines.append(f"  A better than B: {win_rate_A:.3f}")
        report_lines.append(f"  B better than A: {win_rate_B:.3f}")
        report_lines.append(f"  Tie: {tie_rate:.3f}")

    return "\n".join(report_lines)
