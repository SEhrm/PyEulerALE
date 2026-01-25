#!/usr/bin/env python

"""Computes the responses of the transfer function from pitch angle to coefficient of lift/moment

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

import gzip
from argparse import ArgumentParser

import numpy as np

from py_euler_ale import HEAT_RATIO
from py_euler_ale import SpatialDiscretization

parser = ArgumentParser(
    description="Computes the responses of Edwards' transfer function at zero angle-of-attack.")
parser.add_argument("mesh_file", type=str, help="Mesh file.")
parser.add_argument("chord", type=float, help="Chord length (in grid units).")
parser.add_argument("mach_number", type=float, help="Free-stream Mach number.")
parser.add_argument("axis_location", type=float,
                    help="Axis of rotation location (in grid units).")
parser.add_argument(
    "--rtol", type=float, default=1e-9,
    help="Residual tolerance to reach relative to free-stream residual. (default: %(default)s)")
parser.add_argument(
    "--iter", type=int, default=100,
    help="Maximum number of PTC iterations. (default: %(default)s)")
args = parser.parse_args()


def export_pressure_coefficients(
    file_name: str,
    vertices: np.ndarray,
    pressure_coef: np.ndarray,
    pressure_coef_wrt_aoa: np.ndarray,
) -> None:
    """Exports the pressure coefficients for plotting.

    For each cell, separated by an empty line, writes four lines for each cell's vertex with columns
    '`x y cₚ Re(ℒcₚ/ℒα) Im(ℒcₚ/ℒα)`', where `cₚ` is the steady-state pressure coefficient and
    `ℒcₚ/ℒα` is the transfer from pitch angle to pressure coefficient. File is compressed with Gzip.

    Args:
        file_name: File name to write to.
        vertices: Grid vertices.
        pressure_coef: Steady-state pressure coefficient.
        pressure_coef_wrt_aoa: Response from pitch angle to pressure coefficient.
    """
    with gzip.open(file_name, mode="w") as file:
        for m, n in np.ndindex(pressure_coef.shape):
            cell_vertices = np.vstack((
                vertices[:, m, n], vertices[:, m, n + 1],
                vertices[:, m + 1, n + 1], vertices[:, m + 1, n],
            ))
            cell_data = np.tile(np.hstack((
                pressure_coef[m, n],
                pressure_coef_wrt_aoa[m, n].real,
                pressure_coef_wrt_aoa[m, n].imag,
            )), 4).reshape(4, -1)
            # noinspection PyTypeChecker
            np.savetxt(file, np.hstack((cell_vertices, cell_data)), fmt="%+.5e")
            file.write(b"\n")


print(f"{'Mesh file:':<25} {args.mesh_file}")
print(f"{'Chord / L:':<25} {args.chord}")
print(f"{'Mach number:':<25} {args.mach_number}")
print(f"{'Axis location / L:':<25} {args.axis_location}")

# Non-dimensional free-stream speed `vₒₒ/√(pₒₒ/ϱₒₒ) = Maₒₒ⋅√γ`
free_stream_speed = args.mach_number * HEAT_RATIO**0.5

# Initialize solver
solver = SpatialDiscretization(
    grid_file=args.mesh_file,
    mach_number=args.mach_number,
    angle_of_attack=0.,
    coefficient_length=args.chord,
    coefficient_center=(args.axis_location, 0.0),
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
    print(f"{nt:>6} {norm:>15.1e} {rel_norm:>15.1e}")

    # Linearize the solver based on the current ``solver.states``
    solver.linearize()
    solver.compute_forces()

    # If satisfied, break.
    if rel_norm <= args.rtol or np.isnan(rel_norm):
        # Solver is now linearized around the steady-state
        break

    # Get the global pseudo time-step size by switched evolution relaxation (SER)
    time_step_size = 1.e-1 * rel_norm**-1.

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

    # Compute transfer from pitch angle to states
    states_wrt_aoa = -solver.solve_odes_wrt_states_fwd(
        shift=laplace,
        d_odes=(
            solver.apply_odes_wrt_vertices_fwd(
                d_vertices=vertices_wrt_aoa,
            ) +
            solver.apply_odes_wrt_velocities_fwd(
                d_velocities=laplace * vertices_wrt_aoa,
            )
        ),
    )

    # Compute transfer from pitch angle to non-dimensional force
    forces_wrt_aoa = (
        solver.apply_forces_wrt_vertices_fwd(
            d_vertices=vertices_wrt_aoa,
        ) +
        solver.apply_forces_wrt_states_fwd(
            d_states=states_wrt_aoa,
        )
    )

    # Compute transfer from pitch angle to coefficients
    lift_coef_wrt_aoa = np.vdot(
        solver.compute_lift_coefficient_wrt_forces(), forces_wrt_aoa,
    )
    moment_coef_wrt_aoa = np.vdot(
        solver.compute_moment_coefficient_wrt_forces(), forces_wrt_aoa,
    ) + np.vdot(
        solver.compute_moment_coefficient_wrt_vertices(), vertices_wrt_aoa,
    )

    # Print coefficients
    print(f"{reduced_frequency:>15.3e} {lift_coef_wrt_aoa:>+25.3e} {moment_coef_wrt_aoa:>+25.3e}")

# Export steady-state pressure coefficients and transfer from pitch angle to pressure coefficients
# at the last reduced frequency
# noinspection PyUnboundLocalVariable
export_pressure_coefficients(
    file_name="cp.gz",
    vertices=solver.vertices,
    pressure_coef=solver.compute_pressure_coefficients(),
    pressure_coef_wrt_aoa=solver.apply_pressure_coefficients_wrt_states_fwd(states_wrt_aoa),
)
