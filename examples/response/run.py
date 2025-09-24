#!/usr/bin/env python

"""Computes the responses of the transfer function from pitch angle to coefficient of lift/moment

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

from argparse import ArgumentParser

import numpy as np

from py_euler_ale import HEAT_RATIO, SpatialDiscretization

parser = ArgumentParser(
    description="Computes the responses of Edwards' transfer function at zero angle-of-attack.")
parser.add_argument("mesh_file", type=str, help="Mesh file.")
parser.add_argument("chord", type=float, help="Chord length (in grid units).")
parser.add_argument("mach_number", type=float, help="Free-stream Mach number.")
parser.add_argument("axis_location", type=float,
                    help="Axis of rotation location (in grid units).")
parser.add_argument(
    "--rusanov", type=float, default=2e-2,
    help="Rusanov/Lax-Friedrich flux factor, typically between 0 and 1. (default: %(default)s)")
parser.add_argument(
    "--rtol", type=float, default=1e-12,
    help="Residual tolerance to reach relative to free-stream residual. (default: %(default)s)")
parser.add_argument(
    "--iter", type=int, default=100,
    help="Maximum number of PTC iterations. (default: %(default)s)")
args = parser.parse_args()


def coefficients(forces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Computes the section lift coefficient and section moment coefficient

    Reference point for the moment are the grid-coordinates `(0,0)`.

    Args:
        forces: Surface section forces.
        points: Surface points.

    Returns:
        Section lift coefficient and section moment coefficient.
    """
    dynamic_pressure = free_stream_speed**2 / 2  # per free-stream pressure
    points_x, points_z = points
    forces_x, forces_z = forces
    lift_coef = np.sum(forces_z) / (dynamic_pressure * args.chord)
    moment_coef = \
        np.sum(points_z * forces_x - points_x * forces_z) / (dynamic_pressure * args.chord**2)
    return np.array([lift_coef, moment_coef])


print(f"{'Mesh file:':<25} {args.mesh_file}")
print(f"{'Chord / L:':<25} {args.chord}")
print(f"{'Mach number:':<25} {args.mach_number}")
print(f"{'Axis location / L:':<25} {args.axis_location}")
print(f"{'Rusanov flux factor:':<25} {args.rusanov}")

# Non-dimensional free-stream speed `v₀₀/√(p₀₀/ϱ₀₀) = Ma₀₀⋅√γ`
free_stream_speed = args.mach_number * HEAT_RATIO**0.5

# Initialize solver
solver = SpatialDiscretization(
    grid_file=args.mesh_file,
    mach_numer=args.mach_number,
    angle_of_attack=0.,
    rusanov_factor=args.rusanov,
)

# Compute ``solver.odes`` based on ``solver.states`` which is initialized by free-stream
solver.compute_odes()

# Compute the ∞-norm of ``solver.odes`` associated to the free-stream state
norm_free_stream = np.max(np.abs(solver.odes))

# Run pseudo-transient continuation (PTC)
print(f"\nPseudo-transient continuation:\n{'it':>6} {'residual':>15} {'rel residual':>15}")
for nt in range(args.iter):

    # Compute current ``solver.odes`` based on current ``solver.states``
    solver.compute_odes()

    # Compute the ∞-norm of ``solver.odes``
    norm = np.max(np.abs(solver.odes))

    # Compute relative norm
    rel_norm = norm / norm_free_stream

    # Print the current iterate
    print(f"{nt:>6} {norm:>15.3e} {rel_norm:>15.3e}")

    # Linearize the solver based on the current ``solver.states``
    solver.linearize()
    solver.compute_forces()

    # If satisfied, break.
    if rel_norm <= args.rtol or np.isnan(rel_norm):
        # Solver is now linearized around the steady-state
        break

    # Get the global pseudo time-step size by switched evolution relaxation (SER)
    time_step_size = 1. * rel_norm**-1.

    # Solve linearized system for the update based on the time-step size and the current residual
    update = solver.solve_odes_wrt_states_fwd(
        d_odes=-solver.odes, shift=time_step_size**-1.,
    )

    # Update ``solver.states``
    solver.states[:] += update

# Compute transfer from pitch angle to grid vertices (1/rad):
vertices_wrt_aoa = np.empty_like(solver.vertices, dtype=complex)
vertices_wrt_aoa[1] = -(solver.vertices[0] - args.axis_location)
vertices_wrt_aoa[0] = solver.vertices[1]

# Compute transfer from pitch angle to surface points (1/rad):
surface_points_wrt_aoa = np.empty_like(solver.surface_points, dtype=float)
surface_points_wrt_aoa[1] = -(solver.surface_points[0] - args.axis_location)
surface_points_wrt_aoa[0] = solver.surface_points[1]

# Run frequency response
print(f"\nFrequency response:\n"
      f"{'red frequency':>15} {'lift coefficient':>25} {'moment coefficient':>25}")
for reduced_frequency in np.logspace(-2, 0, 11):
    # Non-dimensional complex laplace from reduced frequency
    laplace = 1j * reduced_frequency * free_stream_speed / (args.chord / 2)

    # Compute transfer from pitch angle to non-dimensional force
    forces_wrt_aoa = (
        solver.apply_forces_wrt_vertices_fwd(
            d_vertices=vertices_wrt_aoa,
        ) +
        solver.apply_forces_wrt_states_fwd(
            d_states=-solver.solve_odes_wrt_states_fwd(
                shift=laplace,
                d_odes=(
                    solver.apply_odes_wrt_vertices_fwd(
                        d_vertices=vertices_wrt_aoa,
                    ) +
                    solver.apply_odes_wrt_velocities_fwd(
                        d_velocities=laplace * vertices_wrt_aoa,
                    )
                ),
            ),
        )
    )

    # Compute transfer from pitch angle to coefficients
    lift_coef_wrt_aoa, moment_coef_wrt_aoa = (
        coefficients(forces_wrt_aoa, solver.surface_points - [[args.axis_location], [0]]) +
        coefficients(solver.forces, surface_points_wrt_aoa)
    )

    # Print coefficients
    print(f"{reduced_frequency:>15.3e} {lift_coef_wrt_aoa:>+25.3e} {moment_coef_wrt_aoa:>+25.3e}")
