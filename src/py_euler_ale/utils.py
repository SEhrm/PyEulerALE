"""Utility functions

This module implements utility functions for post-processing.

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

import numpy as np

from .core import HEAT_RATIO


def get_coefficients(
    forces: np.ndarray,
    surface_points: np.ndarray,
    dynamic_pressure: float,
    chord: float,
) -> np.ndarray:
    """Gets the section lift coefficient and section moment coefficient

    Reference point for the moment are the grid-coordinates `(0,0)`.

    Args:
        forces: Surface section forces array in shape ``(NUM_DIM,num_angular)``.
        surface_points: Surface points array in shape ``(NUM_DIM,num_angular)``.
        dynamic_pressure: The dynamic pressure (per free-stream pressure).
        chord: Chord (per grid unit).

    Returns:
        Section lift coefficient and section moment coefficient.
    """
    points_x, points_z = surface_points
    forces_x, forces_z = forces
    lift_coef = np.sum(forces_z) / (dynamic_pressure * chord)
    moment_coef = np.sum(points_z * forces_x - points_x * forces_z) / (dynamic_pressure * chord**2)
    return np.array([lift_coef, moment_coef])


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
