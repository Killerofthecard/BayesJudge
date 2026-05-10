"""
Beta coefficient convergence curve visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from .base_plotter import BasePlotter


class BetaConvergencePlotter(BasePlotter):
    """
    Plots the convergence trajectory of the LLM bias coefficient Beta.
    """

    def __init__(self, max_samples: int, figsize: Tuple[int, int] = (10, 6)):
        super().__init__(max_samples, figsize)
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlabel('Sample Index', fontsize=11)
        self.ax.set_ylabel('LLM Bias Coefficient β (Posterior Mean)', fontsize=11)
        self.ax.set_title('LLM Position Bias Coefficient β Convergence Trajectory', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_xlim(0, max_samples)
        self.ax.set_ylim(-2.0, 2.0)
        self.ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No-bias line (β=0)')

        plt.tight_layout()
        plt.show(block=False)

        self.lines_beta = {}
        print("[Beta Plotter] Initialization complete")

    def update(self, sample_idx: int, evaluator):
        """Update the Beta coefficient plot for the given sample.
        
        Args:
            sample_idx: Current sample index.
            evaluator: Evaluator object providing particle data.
        """
        if not hasattr(evaluator, 'beta_particles') or not evaluator.beta_particles:
            return

        for llm_id in evaluator.beta_particles.keys():
            particles = evaluator.beta_particles[llm_id]
            weights = evaluator.weights.get(llm_id, np.ones(len(particles)) / len(particles))
            
            mean_beta = np.average(particles, weights=weights)
            
            if llm_id not in self.data_history:
                self.data_history[llm_id] = {'x': [], 'beta': []}
                color = self._get_color(llm_id)
                line_beta, = self.ax.plot([], [], 'o-', color=color, label=f'{llm_id}',
                                        markersize=3, linewidth=2, alpha=0.8)
                self.lines_beta[llm_id] = line_beta
                self.ax.legend(loc='upper right', fontsize=9)

            self.data_history[llm_id]['x'].append(sample_idx)
            self.data_history[llm_id]['beta'].append(mean_beta)

            x_data = self.data_history[llm_id]['x']
            beta_data = self.data_history[llm_id]['beta']
            self.lines_beta[llm_id].set_data(x_data, beta_data)

            if beta_data:
                min_beta, max_beta = min(beta_data), max(beta_data)
                margin = max(abs(min_beta), abs(max_beta), 0.5) * 1.2
                self.ax.set_ylim(-margin, margin)

        self._draw()

    def finalize(self, output_path: Optional[str] = None):
        """Save the figure and close the plot.
        
        Args:
            output_path: Optional file path to save the figure.
        """
        for llm_id, data in self.data_history.items():
            if data['x']:
                self.lines_beta[llm_id].set_data(data['x'], data['beta'])
        
        self.fig.canvas.draw()
        
        if output_path:
            self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[Beta Plot] Saved to: {output_path}")
        
        print("\n[Beta Visualization Summary]")
        for llm_id, data in self.data_history.items():
            if data['beta']:
                final_beta = data['beta'][-1]
                print(f"  {llm_id}: final β={final_beta:.4f}, "
                      f"range=[{min(data['beta']):.4f}, {max(data['beta']):.4f}]")
