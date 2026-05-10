"""
Base plotter class for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class BasePlotter:
    """Base class for plotters."""
    
    def __init__(self, max_samples: int, figsize: Tuple[int, int]):
        """
        Initialize the plotter.
        
        Args:
            max_samples: Maximum number of samples.
            figsize: Figure size (width, height).
        """
        self.max_samples = max_samples
        self.data_history: Dict[str, Dict] = {}
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        self.fig = None
        self.axes = None
    
    def _get_color(self, llm_id: str) -> str:
        """Return a color for the given LLM identifier.
        
        Args:
            llm_id: LLM identifier string.
        
        Returns:
            Color string.
        """
        idx = hash(llm_id) % len(self.colors)
        return self.colors[idx]
    
    def update(self, sample_idx: int, evaluator):
        """Update the plot.
        
        Args:
            sample_idx: Current sample index.
            evaluator: Evaluator object.
        
        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError
    
    def finalize(self, output_path: str = None):
        """Finalize the plot.
        
        Args:
            output_path: Optional file path to save the figure.
        
        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError
    
    def _draw(self):
        """Redraw the figure canvas."""
        if self.fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
