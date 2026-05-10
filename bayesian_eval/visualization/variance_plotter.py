"""
Variance and ESS convergence curve visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from .base_plotter import BasePlotter


class VarianceConvergencePlotter(BasePlotter):
    """
    Visualizes posterior variance convergence (left) and effective sample size (ESS) monitoring (right).
    """

    def __init__(self, max_samples: int, figsize: Tuple[int, int] = (12, 6)):
        super().__init__(max_samples, figsize)
        
        self.fig, self.axes = plt.subplots(1, 2, figsize=figsize)
        self.ax_var, self.ax_ess = self.axes
        
        self.ax_var.set_xlabel('Sample Index', fontsize=11)
        self.ax_var.set_ylabel('LLM Bias Posterior Variance', fontsize=11)
        self.ax_var.set_title('LLM Position Bias Variance Convergence Curve', fontsize=12, fontweight='bold')
        self.ax_var.grid(True, alpha=0.3, linestyle='--')
        self.ax_var.set_xlim(0, max_samples)
        self.ax_var.set_ylim(0, 2.0)

        self.ax_ess.set_xlabel('Sample Index', fontsize=11)
        self.ax_ess.set_ylabel('Effective Sample Size (ESS)', fontsize=11)
        self.ax_ess.set_title('Particle Filter Health Monitor (ESS)', fontsize=12, fontweight='bold')
        self.ax_ess.grid(True, alpha=0.3, linestyle='--')
        self.ax_ess.set_xlim(0, max_samples)
        self.ax_ess.set_ylim(0, 1000)

        plt.tight_layout()
        plt.show(block=False)

        self.lines_var = {}
        self.lines_ess = {}
        self.M = 1000
        
        print("[Variance Plotter] Initialization complete")

    def set_particle_count(self, M: int):
        """Set the particle count and adjust the ESS plot y-axis limit.
        
        Args:
            M: Number of particles.
        """
        self.M = M
        self.ax_ess.set_ylim(0, M * 1.1)

    def update(self, sample_idx: int, evaluator):
        """Update variance and ESS plots for the given sample.
        
        Args:
            sample_idx: Current sample index.
            evaluator: Evaluator object providing posterior statistics.
        """
        if not hasattr(evaluator, 'beta_particles') or not evaluator.beta_particles:
            return

        for llm_id in evaluator.beta_particles.keys():
            stats = evaluator.get_llm_posterior_stats(llm_id)
            if not stats:
                continue

            variance = stats.get('variance')
            ess = stats.get('ess')
            if variance is None or ess is None:
                continue

            if llm_id not in self.data_history:
                self.data_history[llm_id] = {'x': [], 'variance': [], 'ess': []}
                color = self._get_color(llm_id)
                line_var, = self.ax_var.plot([], [], 'o-', color=color, label=f'{llm_id}',
                                           markersize=3, linewidth=1.5, alpha=0.8)
                line_ess, = self.ax_ess.plot([], [], 's-', color=color, label=f'{llm_id}',
                                           markersize=3, linewidth=1.5, alpha=0.8)
                self.lines_var[llm_id] = line_var
                self.lines_ess[llm_id] = line_ess
                self.ax_var.legend(loc='upper right', fontsize=8)
                self.ax_ess.legend(loc='lower right', fontsize=8)

            self.data_history[llm_id]['x'].append(sample_idx)
            self.data_history[llm_id]['variance'].append(variance)
            self.data_history[llm_id]['ess'].append(ess)

            x_data = self.data_history[llm_id]['x']
            self.lines_var[llm_id].set_data(x_data, self.data_history[llm_id]['variance'])
            self.lines_ess[llm_id].set_data(x_data, self.data_history[llm_id]['ess'])

            if variance > self.ax_var.get_ylim()[1] * 0.9:
                self.ax_var.set_ylim(0, max(variance * 1.2, 2.0))
            if ess > self.ax_ess.get_ylim()[1] * 0.9:
                self.ax_ess.set_ylim(0, min(ess * 1.2, self.M * 1.1))

        self._draw()

    def finalize(self, output_path: Optional[str] = None):
        """Save the figure and close the plot.
        
        Args:
            output_path: Optional file path to save the figure.
        """
        for llm_id, data in self.data_history.items():
            if data['x']:
                self.lines_var[llm_id].set_data(data['x'], data['variance'])
                self.lines_ess[llm_id].set_data(data['x'], data['ess'])
        
        self.fig.canvas.draw()
        
        if output_path:
            self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[Variance Plot] Saved to: {output_path}")
        
        print("\n[Variance Visualization Summary]")
        for llm_id, data in self.data_history.items():
            if data['variance']:
                print(f"  {llm_id}: final variance={data['variance'][-1]:.6f}, "
                      f"range=[{min(data['variance']):.6f}, {max(data['variance']):.6f}]")
