"""Utility functions

This module implements utility functions for post-processing.

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

import numpy as np

from .core import HEAT_RATIO


def get_pressure(state: np.ndarray) -> np.number | np.ndarray:
    """Gets the pressure from the four conserved variables

    Args:
        state: Conserved variables array in shape ``(NUM_VAR)`` or
            ``(NUM_VAR,num_radial,num_angular)``.

    Returns:
        Pressure as scalar or as array in shape ``(num_radial,num_angular)``.
    """
    density, momentum_density_x, momentum_density_z, total_energy_density = state
    return (HEAT_RATIO - 1.) * (
        total_energy_density - (momentum_density_x**2 + momentum_density_z**2) / 2. / density
    )


def get_pressure_derivative(state: np.ndarray, d_state: np.ndarray) -> np.number | np.ndarray:
    """Gets the pressure derivative from the four conserved variables and their derivatives.

    Args:
        state: Conserved variables array in shape ``(NUM_VAR)`` or
            ``(NUM_VAR,num_radial,num_angular)``.
        d_state: Conserved variables derivative array in shape ``(NUM_VAR)`` or
            ``(NUM_VAR,num_radial,num_angular)``.

    Returns:
        Pressure derivative as scalar or as array in shape ``(num_radial,num_angular)``.
    """
    density, momentum_density_x, momentum_density_z, _ = state
    d_density, d_momentum_density_x, d_momentum_density_z, d_total_energy_density = d_state
    return (HEAT_RATIO - 1.) * (
        d_total_energy_density +
        (momentum_density_x**2 + momentum_density_z**2) / density**2 / 2. * d_density -
        (
            momentum_density_x * d_momentum_density_x + momentum_density_z * d_momentum_density_z
        ) / density
    )
