"""Utility functions

This module implements utility functions for post-processing.

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

import numpy as np

from .core import HEAT_RATIO


def get_section_coefficients(
    forces: np.ndarray,
    surface_points: np.ndarray,
    mach_number: float,
    chord: float,
) -> np.ndarray:
    """Gets the section lift, drag and moment coefficient

    Reference point for the moment are the grid-coordinates `(0,0)`.

    Args:
        forces: Surface section forces array in shape ``(NUM_DIM,num_angular)``.
        surface_points: Surface points array in shape ``(NUM_DIM,num_angular)``.
        mach_number: Free-stream Mach number.
        chord: Chord.

    Returns:
        Section lift, drag and moment coefficient.
    """
    dynamic_pressure = mach_number**2 * HEAT_RATIO / 2
    points_x, points_z = surface_points
    forces_x, forces_z = forces
    drag_coef = np.sum(forces_x) / (dynamic_pressure * chord)
    lift_coef = np.sum(forces_z) / (dynamic_pressure * chord)
    moment_coef = np.sum(points_z * forces_x - points_x * forces_z) / (dynamic_pressure * chord**2)
    return np.array([drag_coef, lift_coef, moment_coef])


def get_pressure_coefficient(
    state: np.ndarray,
    mach_number: float,
) -> np.number | np.ndarray:
    """Gets the pressure coefficient

    Args:
        state: Conserved variables array in shape ``(NUM_VAR)`` or
            ``(NUM_VAR,num_radial,num_angular)``.
        mach_number: Free-stream Mach number.

    Returns:
        Pressure coefficient as scalar or as array in shape ``(num_radial,num_angular)``.
    """
    dynamic_pressure = mach_number**2 * HEAT_RATIO / 2
    density, momentum_density_x, momentum_density_z, total_energy_density = state
    pressure = (HEAT_RATIO - 1.) * (
        total_energy_density - (momentum_density_x**2 + momentum_density_z**2) / 2. / density
    )
    return (pressure - 1.) / dynamic_pressure


def get_pressure_coefficient_derivative(
    state: np.ndarray,
    d_state: np.ndarray,
    mach_number: float,
) -> np.number | np.ndarray:
    """Gets the pressure coefficient derivative

    Args:
        state: Conserved variables array in shape ``(NUM_VAR)`` or
            ``(NUM_VAR,num_radial,num_angular)``.
        d_state: Conserved variables derivative array in shape ``(NUM_VAR)`` or
            ``(NUM_VAR,num_radial,num_angular)``.
        mach_number: Free-stream Mach number (kept constant).

    Returns:
        Pressure coefficient derivative as scalar or as array in shape ``(num_radial,num_angular)``.
    """
    dynamic_pressure = mach_number**2 * HEAT_RATIO / 2
    density, momentum_density_x, momentum_density_z, _ = state
    d_density, d_momentum_density_x, d_momentum_density_z, d_total_energy_density = d_state
    d_pressure = (HEAT_RATIO - 1.) * (
        d_total_energy_density +
        (momentum_density_x**2 + momentum_density_z**2) / density**2 / 2. * d_density -
        (
            momentum_density_x * d_momentum_density_x + momentum_density_z * d_momentum_density_z
        ) / density
    )
    return d_pressure / dynamic_pressure
