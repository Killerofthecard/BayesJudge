import json
import numpy as np
from scipy.special import softmax
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path

@dataclass
class SampleData:
    """
    Data structure for evaluation samples.

    Attributes:
        human_judgments: Human annotations Y_t^H = {y_{tr}^H}, format {rater_id: label}
        llm_judgments: LLM swap annotations Y_tj^L, format {llm_id: (label_AB, label_BA)}
        metadata: Optional metadata (question_id, model_a, model_b, etc.)
        conversation_a: Dialogue content for model A
        conversation_b: Dialogue content for model B
    """
    human_judgments: Dict[str, str]
    llm_judgments: Dict[str, Tuple[str, str]]
    metadata: Optional[Dict] = None
    conversation_a: Optional[List[Dict[str, str]]] = None
    conversation_b: Optional[List[Dict[str, str]]] = None

    def get_question_id(self) -> str:
        """Get the question ID."""
        if self.metadata and 'question_id' in self.metadata:
            return self.metadata['question_id']
        return 'N/A'

    def has_llm_judgment(self, llm_id: str) -> bool:
        """Check if a judgment from the specified LLM exists."""
        return llm_id in self.llm_judgments

    def get_llm_judgment(self, llm_id: str) -> Optional[Tuple[str, str]]:
        """Get the judgment from the specified LLM."""
        return self.llm_judgments.get(llm_id)

    def to_dict(self) -> dict:
        """Convert to dictionary (for JSON serialization)."""
        return asdict(self)


class SyntheticDataGenerator:
    """
    Synthetic data generator (enhanced).
    Forward-generates observed labels based on true parameters (eta, C_r, beta).
    Supports flexible configuration of multiple human and LLM evaluators.
    Note: uniformly uses 'C' to indicate ties, matching the actual LLM implementation.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.label_map = {0: 'A', 1: 'C', 2: 'B'}
        self.idx_map = {'A': 0, 'C': 1, 'B': 2}

        self.human_raters = []
        self.llm_judges = []

    def setup_ground_truth(
        self,
        n_samples: int = 500,
        human_raters: Optional[List[Dict]] = None,
        llm_judges: Optional[List[Dict]] = None,
        class_ratios: Optional[Dict[str, float]] = None
    ):
        """
        Set the ground-truth parameters (God's-eye view).

        Parameters:
            n_samples: Number of samples
            human_raters: List of human rater configurations, each a dict:
                {
                    'name': 'rater_0',
                    'confusion_matrix': [[...],
                                         [...],
                                         [...]]
                    'description': 'Optional description'
                }
            llm_judges: List of LLM judge configurations, each a dict:
                {
                    'name': 'llm_0',
                    'beta': 0.5,
                    'description': 'Optional description'
                }
            class_ratios: Control for three-class sample ratios, format {'A': 0.4, 'B': 0.4, 'C': 0.2}
                         where 'A'=A wins, 'B'=B wins, 'C'=tie
                         If None, defaults to {'A': 0.4, 'B': 0.4, 'C': 0.2}
        """
        self.n_samples = n_samples

        if class_ratios is None:
            self.class_ratios = {'A': 0.4, 'B': 0.4, 'C': 0.2}
        else:
            total = sum(class_ratios.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Class ratios must sum to 1.0, got {total}")
            self.class_ratios = class_ratios.copy()

        n_a = int(n_samples * self.class_ratios['A'])
        n_b = int(n_samples * self.class_ratios['B'])
        n_tie = n_samples - n_a - n_b

        group_a = np.column_stack([
            np.random.normal(2.0, 0.5, n_a),
            np.random.normal(0.1, 0.05, n_a)
        ])

        group_b = np.column_stack([
            np.random.normal(-2.0, 0.5, n_b),
            np.random.normal(0.1, 0.05, n_b)
        ])

        group_tie = np.column_stack([
            np.random.normal(0.0, 0.3, n_tie),
            np.random.normal(1.5, 0.3, n_tie)
        ])

        self.eta_true = np.vstack([group_a, group_b, group_tie])
        np.random.shuffle(self.eta_true)

        if human_raters is None:
            human_raters = [
                {
                    'name': 'rater_0',
                    'confusion_matrix': [
                        [0.95, 0.03, 0.02],
                        [0.05, 0.90, 0.05],
                        [0.02, 0.03, 0.95]
                    ],
                    'description': 'Perfect rater'
                },
                {
                    'name': 'rater_1',
                    'confusion_matrix': [
                        [0.90, 0.06, 0.04],
                        [0.25, 0.65, 0.10],
                        [0.20, 0.05, 0.75]
                    ],
                    'description': 'Biased rater (high probability of misjudging C/B as A)'
                }
            ]

        self.human_raters = human_raters
        self.C_true = [np.array(r['confusion_matrix']) for r in human_raters]

        if llm_judges is None:
            llm_judges = [
                {
                    'name': 'llm_judge',
                    'accuracy': 0.8,
                    'beta': 0.5,
                    'description': 'Standard LLM (beta=0.5, strict softmax sampling)'
                }
            ]

        self.llm_judges = llm_judges
        self.beta_true = [llm.get('beta', 0.5) for llm in self.llm_judges]

        print(f"[Ground Truth] Generated true parameters for {n_samples} samples:")
        print(f"  - Class ratios: A={self.class_ratios['A']:.0%}, B={self.class_ratios['B']:.0%}, C={self.class_ratios['C']:.0%}")
        print(f"  - A-win samples: {n_a}, B-win samples: {n_b}, Tie samples: {n_tie}")
        print(f"  - Label convention: A=advantaged, C=Tie, B=disadvantaged")
        print(f"\n  Human rater configuration ({len(self.human_raters)}):")
        for i, rater in enumerate(self.human_raters):
            print(f"    - {rater['name']}: {rater.get('description', 'No description')}")
        print(f"\n  LLM judge configuration ({len(self.llm_judges)}):")
        for i, llm in enumerate(self.llm_judges):
            beta = llm.get('beta', 0.5)
            print(f"    - {llm['name']}: beta={beta:.2f}")

    def _compute_pi(self, theta: float, tau: float) -> np.ndarray:
        """Compute the true verdict distribution pi_t = softmax([theta, tau, -theta])."""
        logits = np.array([theta, tau, -theta])
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def _sample_human_labels(self, true_label_idx: int) -> Dict[str, str]:
        """
        Sample human labels based on the true label index (consistent with confusion matrix definition).

        Parameters:
            true_label_idx: 0=A, 1=C (tie), 2=B
        Returns:
            {rater_name: 'A'/'C'/'B'}
        """
        labels = {}
        for rater_config, C_r in zip(self.human_raters, self.C_true):
            rater_name = rater_config['name']
            probs = C_r[true_label_idx, :]
            probs = probs / probs.sum()

            label_idx = np.random.choice([0, 1, 2], p=probs)
            labels[rater_name] = self.label_map[label_idx]
        return labels

    def _sample_llm_swap_strict(self, theta_t: float, tau_t: float, beta_j: float) -> Tuple[str, str]:
        """
        Strictly follow the swap-design likelihood function in Section 2.2.2 of the paper.

        Parameters:
            theta_t: Quality difference parameter (degree to which A is better than B; positive means A is better)
            tau_t: Tie quality parameter (larger means higher tie probability)
            beta_j: LLM position bias coefficient (strength of first-position preference)

        Returns:
            (AB-order judgment, BA-order judgment), where 'A'/'C'/'B' indicates the choice

        Paper formula notes:
        - AB-order logits: [theta + beta, tau, -theta - beta] -> [A, Tie, B]
        - BA-order logits: [theta - beta, tau, -theta + beta] -> [A, Tie, B]
        Note: 'A' means option A, 'B' means option B, independent of presentation order
        """
        logits_ab = np.array([theta_t + beta_j, tau_t, -theta_t - beta_j])

        logits_ba = np.array([-theta_t + beta_j, tau_t, theta_t - beta_j])

        probs_ab = softmax(logits_ab)
        probs_ba = softmax(logits_ba)

        labels = ['A', 'C', 'B']
        y_ab = labels[np.random.choice(3, p=probs_ab)]
        y_ba = labels[np.random.choice(3, p=probs_ba)]

        return y_ab, y_ba

    def _sample_llm_judgments(self, sample_idx: int) -> Dict[str, Tuple[str, str]]:
        """
        Sample judgments from all LLMs based on the sample index (using strict paper formulas).

        Parameters:
            sample_idx: Sample index, used to retrieve theta_t and tau_t for this sample
        Returns:
            {llm_name: (AB-order judgment, BA-order judgment)}
        """
        theta_t, tau_t = self.eta_true[sample_idx]

        judgments = {}
        for llm_config, beta_j in zip(self.llm_judges, self.beta_true):
            name = llm_config['name']
            judgments[name] = self._sample_llm_swap_strict(theta_t, tau_t, beta_j)

        return judgments

    def _sample_llm_judgments_perfect(self, sample_idx: int) -> Dict[str, Tuple[str, str]]:
        """
        Perfect LLM judgments: generate answers consistent with the true label regardless of presentation order.

        This function simulates a "perfect" LLM evaluator that:
        1. Identifies ground truth with 100% accuracy, unaffected by position bias
        2. Gives the same correct answer under both AB and BA orders
        3. Used to test the Bayesian evaluator under ideal conditions

        Parameters:
            sample_idx: Sample index, used to retrieve the true_label for this sample
        Returns:
            {llm_name: (AB judgment, BA judgment)}, where AB and BA both point to the correct option
        """
        theta_t, tau_t = self.eta_true[sample_idx]
        pi_true = self._compute_pi(theta_t, tau_t)
        true_label_idx = np.argmax(pi_true)
        true_label = self.label_map[true_label_idx]

        judgments = {}
        for llm_config in self.llm_judges:
            name = llm_config['name']
            if true_label == "A":
                judgments[name] = ('A', 'B')
            elif true_label == "B":
                judgments[name] = ('B', 'A')
            elif true_label == "C":
                judgments[name] = ('C', 'C')

        return judgments

    def generate_dataset(self, save_path: Optional[str] = None,
                        perfect_llm: bool = False) -> List[SampleData]:
        """
        Generate the complete synthetic dataset.

        Parameters:
            save_path: Save path
            perfect_llm: If True, use the perfect LLM judgment function (always returns true label)
                        If False, use the default noisy LLM judgment function
        """
        dataset = []

        llm_judge_func = self._sample_llm_judgments_perfect if perfect_llm else self._sample_llm_judgments

        for t in range(self.n_samples):
            theta, tau = self.eta_true[t]

            pi_true = self._compute_pi(theta, tau)
            true_label_idx = np.argmax(pi_true)
            true_label = self.label_map[true_label_idx]

            human_labels = self._sample_human_labels(true_label_idx)
            llm_judgments = llm_judge_func(t)

            metadata = {
                'question_id': f'synthetic_{t:04d}',
                'true_theta': float(theta),
                'true_tau': float(tau),
                'true_pi': pi_true.tolist(),
                'true_label': true_label,
                'sample_type': 'A_win' if theta > 1.0 else ('B_win' if theta < -1.0 else 'tie')
            }

            sample = SampleData(
                human_judgments=human_labels,
                llm_judgments=llm_judgments,
                metadata=metadata,
                conversation_a=None,
                conversation_b=None
            )

            dataset.append(sample)

            if (t + 1) % 100 == 0:
                print(f"  Generated {t+1}/{self.n_samples} samples...")

        if save_path:
            self._save_dataset(dataset, save_path, perfect_llm=perfect_llm)

        return dataset

    def _get_config_dict(self, perfect_llm: bool = False) -> dict:
        """
        Get the complete dictionary of current data generation configuration.

        Returns:
            Configuration dictionary containing all generation parameters
        """
        human_rater_list = []
        for i, (rater_config, C_matrix) in enumerate(zip(self.human_raters, self.C_true)):
            human_rater_list.append({
                "id": i,
                "name": rater_config['name'],
                "description": rater_config.get('description', ''),
                "confusion_matrix": C_matrix.tolist()
            })

        llm_list = []
        for i, (llm_config, beta) in enumerate(zip(self.llm_judges, self.beta_true)):
            llm_list.append({
                "id": i,
                "name": llm_config['name'],
                "beta": float(beta),
                "accuracy": llm_config.get('accuracy', 0.8),
                "description": llm_config.get('description', ''),
                "judgment_mode": "perfect" if perfect_llm else "strict_softmax",
                "paper_reference": "Section 2.2.2: swap design likelihood"
            })

        config = {
            "generator_info": {
                "class": "SyntheticDataGenerator",
                "version": "2.0",
                "description": "Synthetic data generator that forward-samples observed labels from ground-truth parameters (supports multi-rater config)"
            },
            "random_seed": 42,
            "sample_config": {
                "n_samples": self.n_samples,
                "label_map": self.label_map,
                "idx_map": self.idx_map
            },
            "class_ratios": {
                "description": "Class ratios (A=A wins, B=B wins, C=Tie)",
                "A_win": self.class_ratios.get('A', 0.4),
                "B_win": self.class_ratios.get('B', 0.4),
                "tie": self.class_ratios.get('C', 0.2)
            },
            "ground_truth_distribution": {
                "description": "Generation of eta_t = (theta, tau)",
                "A_win_group": {
                    "proportion": self.class_ratios.get('A', 0.4),
                    "theta_distribution": "N(2.0, 0.5)",
                    "tau_distribution": "N(0.1, 0.05)",
                    "description": "A wins: theta > 0, tau small (low uncertainty)"
                },
                "B_win_group": {
                    "proportion": self.class_ratios.get('B', 0.4),
                    "theta_distribution": "N(-2.0, 0.5)",
                    "tau_distribution": "N(0.1, 0.05)",
                    "description": "B wins: theta < 0"
                },
                "tie_group": {
                    "proportion": self.class_ratios.get('C', 0.2),
                    "theta_distribution": "N(0.0, 0.3)",
                    "tau_distribution": "N(1.5, 0.3)",
                    "description": "Tie: theta ~ 0, tau large (high uncertainty)"
                }
            },
            "human_rater_config": {
                "description": "Human rater confusion matrices C_r",
                "note": "Rows = true classes [A, C, B], columns = observed classes [A, C, B]",
                "num_raters": len(self.human_raters),
                "raters": human_rater_list
            },
            "llm_config": {
                "description": "LLM judge configuration",
                "num_llms": len(self.llm_judges),
                "llms": llm_list,
                "judgment_mode": "perfect" if perfect_llm else "noisy"
            },
            "label_convention": {
                "A": "Advantage side (A better than B)",
                "C": "Tie",
                "B": "Disadvantage side (B better than A)",
                "note": "Unified use of 'C' to indicate tie, matching actual LLM implementation"
            }
        }
        return config

    def _save_dataset(self, dataset: List[SampleData], save_path: str, perfect_llm: bool = False):
        """
        Save dataset as JSON, along with configuration information.

        Parameters:
            dataset: Dataset
            save_path: Data file save path
            perfect_llm: Whether to use perfect LLM mode
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data_list = [s.to_dict() for s in dataset]

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)

        print(f"\n[Save] Dataset saved to: {save_path}")
        print(f"       Total samples: {len(dataset)}")
        print(f"       Label convention: A=advantaged, C=tie, B=disadvantaged")

        config = self._get_config_dict(perfect_llm)
        config_path = save_path.parent / f"{save_path.stem}_config.json"

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"[Save] Configuration file saved to: {config_path}")
        print(f"       Contains: generation parameters, human prior settings, LLM configuration, etc.")

    def print_statistics(self, dataset: List[SampleData]):
        """Print dataset statistics (supports multiple evaluators)."""
        print("\n" + "="*60)
        print("Synthetic dataset statistics (labels: A/C/B, C is tie)")
        print("="*60)

        true_labels = [s.metadata['true_label'] for s in dataset]
        print("True label distribution:")
        for label in ['A', 'C', 'B']:
            count = true_labels.count(label)
            print(f"  True {label} win: {count} ({count/len(dataset)*100:.1f}%)")

        print(f"\nHuman evaluator statistics ({len(self.human_raters)}):")
        for rater_config in self.human_raters:
            rater_name = rater_config['name']
            correct_count = 0
            for s in dataset:
                true_label = s.metadata['true_label']
                human_label = s.human_judgments.get(rater_name, 'unknown')
                if human_label == true_label:
                    correct_count += 1
            accuracy = correct_count / len(dataset)
            print(f"  {rater_name}: accuracy {accuracy:.1%} ({correct_count}/{len(dataset)})")

        print(f"\nLLM evaluator statistics ({len(self.llm_judges)}):")
        for llm_config in self.llm_judges:
            llm_name = llm_config['name']
            ab_correct = 0
            ba_correct = 0
            conflict_count = 0

            for s in dataset:
                true_label = s.metadata['true_label']
                ab_label, ba_label = s.llm_judgments.get(llm_name, ('unknown', 'unknown'))

                if ab_label == true_label:
                    ab_correct += 1
                if ba_label == true_label:
                    ba_correct += 1
                if (ab_label == 'A' and ba_label == 'B') or (ab_label == 'B' and ba_label == 'A'):
                    conflict_count += 1

            ab_acc = ab_correct / len(dataset)
            ba_acc = ba_correct / len(dataset)
            print(f"  {llm_name}:")
            print(f"    AB-order accuracy: {ab_acc:.1%}")
            print(f"    BA-order accuracy: {ba_acc:.1%}")
            print(f"    Swap conflict samples: {conflict_count} ({conflict_count/len(dataset)*100:.1f}%)")

        print("="*60)


def example_default_config():
    """Example 1: Generate data using default configuration."""
    print("="*70)
    print("Example 1: Using default configuration (2 humans, 1 LLM)")
    print("="*70)

    generator = SyntheticDataGenerator(seed=42)
    generator.setup_ground_truth(n_samples=500)

    n_humans = len(human_raters) if human_raters else 2
    n_llms = len(llm_judges) if llm_judges else 1
    dataset = generator.generate_dataset(
        save_path=f'./dataset/synthetic_500_h{n_humans}l{n_llms}_default.json',
        perfect_llm=False
    )
    generator.print_statistics(dataset)


def example_custom_human_raters():
    """Example 2: Custom multiple human evaluators."""
    print("\n" + "="*70)
    print("Example 2: Custom 3 human evaluators (different bias levels)")
    print("="*70)

    human_raters = [
        {
            'name': 'expert_rater',
            'confusion_matrix': [
                [0.98, 0.01, 0.01],
                [0.02, 0.96, 0.02],
                [0.01, 0.01, 0.98]
            ],
            'description': 'Expert rater (98% accuracy)'
        },
        {
            'name': 'normal_rater',
            'confusion_matrix': [
                [0.90, 0.05, 0.05],
                [0.10, 0.80, 0.10],
                [0.05, 0.05, 0.90]
            ],
            'description': 'Normal rater (avg 87% accuracy)'
        },
        {
            'name': 'noisy_rater',
            'confusion_matrix': [
                [0.70, 0.20, 0.10],
                [0.30, 0.60, 0.10],
                [0.20, 0.20, 0.60]
            ],
            'description': 'Noisy rater (avg 63% accuracy)'
        }
    ]

    generator = SyntheticDataGenerator(seed=42)
    generator.setup_ground_truth(
        n_samples=500,
        human_raters=human_raters,
        class_ratios={'A': 0.5, 'B': 0.3, 'C': 0.2}
    )

    dataset = generator.generate_dataset(
        save_path=f'./dataset/synthetic_500_h{len(human_raters)}l1_3humans.json'
    )
    generator.print_statistics(dataset)


def example_custom_llm_judges():
    """Example 3: Custom multiple LLM evaluators."""
    print("\n" + "="*70)
    print("Example 3: Custom 3 LLM evaluators (different accuracies)")
    print("="*70)

    llm_judges = [
        {
            'name': 'gpt4',
            'beta': 0.2,
            'description': 'GPT-4 level (beta=0.2, low position bias)'
        },
        {
            'name': 'gpt35',
            'beta': 0.6,
            'description': 'GPT-3.5 level (beta=0.6, medium position bias)'
        },
        {
            'name': 'small_llm',
            'beta': 1.2,
            'description': 'Small model (beta=1.2, high position bias)'
        }
    ]

    generator = SyntheticDataGenerator(seed=42)
    generator.setup_ground_truth(
        n_samples=500,
        llm_judges=llm_judges,
        class_ratios={'A': 0.3, 'B': 0.3, 'C': 0.4}
    )

    dataset = generator.generate_dataset(
        save_path=f'./dataset/synthetic_500_h1l{len(llm_judges)}_3llms.json'
    )
    generator.print_statistics(dataset)


def example_full_custom():
    """Example 4: Fully custom configuration (multiple humans + multiple LLMs)."""
    print("\n" + "="*70)
    print("Example 4: Fully custom (2 humans + 2 LLMs)")
    print("="*70)

    human_raters = [
        {
            'name': 'rater_expert',
            'confusion_matrix': [
                [0.90, 0.05, 0.05],
                [0.10, 0.80, 0.10],
                [0.10, 0.05, 0.85]
            ],
            'description': 'Expert-level human rater'
        },
        {
            'name': 'rater_novice',
            'confusion_matrix': [
                [0.50, 0.25, 0.25],
                [0.30, 0.40, 0.30],
                [0.25, 0.25, 0.50]
            ],
            'description': 'Novice-level human rater'
        }

    ]

    llm_judges = [
        {
            'name': 'llm_strong',
            'beta': 0.0,
            'description': 'Strong LLM (beta=0.0, low position bias)'
        },
        {
            'name': 'llm_weak_front_bias',
            'beta': 2.0,
            'description': 'Weak LLM (beta=2.0, high position bias, front-biased)'
        }
    ]

    n_samples = 1000
    generator = SyntheticDataGenerator(seed=42)
    generator.setup_ground_truth(
        n_samples=n_samples,
        human_raters=human_raters,
        llm_judges=llm_judges,
        class_ratios={'A': 0.4, 'B': 0.4, 'C': 0.2}
    )

    n_humans = len(human_raters)
    n_llms = len(llm_judges)
    dataset_perfect = generator.generate_dataset(
        save_path=f'./dataset/synthetic_{n_samples}_h{n_humans}l{n_llms}_custom.json',
        perfect_llm=False
    )

    print("\nPerfect LLM version statistics:")
    generator.print_statistics(dataset_perfect)


def example_compare_sampling_methods():
    """Example: Compare independent sampling and position-bias sampling."""
    print("\n" + "="*70)
    print("Example: Compare two LLM sampling logics (independent sampling vs position bias)")
    print("="*70)

    llm_config = {
        'name': 'test_llm',
        'beta': 0.5,
        'description': 'Test LLM (beta=0.5, strict softmax sampling)'
    }

    generator = SyntheticDataGenerator(seed=42)
    generator.setup_ground_truth(n_samples=1000, llm_judges=[llm_config])

    print("\n[Strict Softmax Sampling] (paper formula)")
    print("  - Using the swap-design likelihood function from Section 2.2.2 of the paper")
    print("  - logits_AB = [theta + beta, tau, -theta - beta]")
    print("  - logits_BA = [theta - beta, tau, -theta + beta]")

    print("\n  Sampling results under different quality parameters:")
    test_cases = [
        (2.0, 0.3, "A significantly better than B"),
        (-2.0, 0.3, "B significantly better than A"),
        (0.0, 1.5, "Tie case"),
        (0.5, 0.5, "A slightly better than B"),
    ]

    for theta, tau, desc in test_cases:
        ab_counts = {'A': 0, 'C': 0, 'B': 0}
        ba_counts = {'A': 0, 'C': 0, 'B': 0}
        for _ in range(1000):
            y_ab, y_ba = generator._sample_llm_swap_strict(theta, tau, beta_j=0.5)
            ab_counts[y_ab] += 1
            ba_counts[y_ba] += 1
        print(f"\n  {desc} (theta={theta}, tau={tau}):")
        print(f"    AB order: A={ab_counts['A']/10:.1f}%, C={ab_counts['C']/10:.1f}%, B={ab_counts['B']/10:.1f}%")
        print(f"    BA order: A={ba_counts['A']/10:.1f}%, C={ba_counts['C']/10:.1f}%, B={ba_counts['B']/10:.1f}%")


def main():
    """Main function: run all examples."""

    example_full_custom()

    print("\n" + "="*70)
    print("All examples completed! Data files and configuration files saved to dataset/ directory")
    print("="*70)


if __name__ == "__main__":
    main()
