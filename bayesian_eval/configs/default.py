"""
Default configuration.
"""

from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    'data_path': '',
    'n_samples': 500,
    'model': 'gpt-4o-mini',
    'n_particles': 500,
    'lambda_diag': 1.0,
    'lambda_off': 0.0,
    'sigma_beta': 1.0,
    'ess_threshold_factor': 0.5,
    'mcmc_step_size': 0.5,
    'verbose': True,
    'print_interval': 1,
    'max_print_raters': 2,
    'live_plot': True,
    'output_dir': './outputs',
}


DEBUG_CONFIG: Dict[str, Any] = {
    **DEFAULT_CONFIG,
    'n_samples': 50,
    'n_particles': 100,
    'live_plot': False,
    'verbose': False,
}


HIGH_PRECISION_CONFIG: Dict[str, Any] = {
    **DEFAULT_CONFIG,
    'n_samples': 1000,
    'n_particles': 2000,
    'lambda_diag': 2.0,
    'lambda_off': 1.0,
}


def get_config(name: str = 'default') -> Dict[str, Any]:
    """
    Retrieve a predefined configuration.
    
    Args:
        name: Configuration name ('default', 'debug', 'high_precision').
    
    Returns:
        A copy of the requested configuration dictionary.
    """
    configs = {
        'default': DEFAULT_CONFIG,
        'debug': DEBUG_CONFIG,
        'high_precision': HIGH_PRECISION_CONFIG,
    }
    
    if name not in configs:
        raise ValueError(f"Unknown configuration: {name}. Available: {list(configs.keys())}")
    
    return configs[name].copy()


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations; values in override take precedence.
    
    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.
    
    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()
    result.update(override)
    return result
