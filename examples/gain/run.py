#!/usr/bin/env python

"""Computes the second derivative of the lift coefficient wrt pitch angle and Mach number.

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

from argparse import ArgumentParser

import numpy as np

from py_euler_ale import HEAT_RATIO
from py_euler_ale import SpatialDiscretization

parser = ArgumentParser(
    description="Computes the lift slope wrt the Mach number at zero angle-of-attack.")
parser.add_argument("mesh_file", type=str, help="Mesh file.")
parser.add_argument("chord", type=float, help="Chord length (in grid units).")
parser.add_argument("mach_number", type=float, help="Free-stream Mach number.")
parser.add_argument(
    "--rtol", type=float, default=1e-9,
    help="Residual tolerance to reach relative to free-stream residual. (default: %(default)s)")
parser.add_argument(
    "--iter", type=int, default=100,
    help="Maximum number of PTC iterations. (default: %(default)s)")
args = parser.parse_args()

print(f"{'Mesh file:':<25} {args.mesh_file}")
print(f"{'Chord / L:':<25} {args.chord}")
print(f"{'Mach number:':<25} {args.mach_number}")

# Non-dimensional dynamic pressure `q₀₀/p₀₀ = Ma₀₀²⋅γ/2`
dynamic_pressure = args.mach_number**2 * HEAT_RATIO / 2

# Initialize solver
solver = SpatialDiscretization(
    grid_file=args.mesh_file,
    mach_number=args.mach_number,
    angle_of_attack=0.,
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

# Grid vertices wrt pitch angle (1/rad)
vertices_wrt_aoa = np.empty_like(solver.vertices, dtype=complex)
vertices_wrt_aoa[1] = -solver.vertices[0]
vertices_wrt_aoa[0] = solver.vertices[1]

# Lift wrt forces gradient
lift_wrt_forces_adj = np.zeros_like(solver.forces, dtype=complex)
lift_wrt_forces_adj[1, :] = 1.

# Compute transfer from pitch angle to states
states_wrt_aoa = -solver.solve_odes_wrt_states_fwd(
    d_odes=solver.apply_odes_wrt_vertices_fwd(d_vertices=vertices_wrt_aoa),
)

# Compute transfer from Mach number to states
states_wrt_mach = -solver.solve_odes_wrt_states_fwd(
    d_odes=solver.apply_odes_wrt_mach_fwd(d_mach=np.asarray([1. + 0j])),
)

# Compute adjoint state gradient
lift_wrt_odes_gradient = -solver.solve_odes_wrt_states_adj(
    solver.apply_forces_wrt_states_rev(d_forces=lift_wrt_forces_adj),
)

# Compute second derivative of lift wrt pitch angle and Mach number
lift_wrt_aoa_wrt_mach = (
    np.vdot(
        solver.compute_forces_wrt_states_inner_product_wrt_states(
            d_forces=lift_wrt_forces_adj, d_states=states_wrt_aoa,
        ), states_wrt_mach,
    ) +
    np.vdot(
        solver.compute_forces_wrt_vertices_inner_product_wrt_states(
            d_forces=lift_wrt_forces_adj, d_vertices=vertices_wrt_aoa,
        ), states_wrt_mach,
    ) +
    solver.compute_odes_wrt_states_inner_product_wrt_mach(
        d_odes=lift_wrt_odes_gradient, d_states=states_wrt_aoa,
    ).item() +
    np.vdot(
        solver.compute_odes_wrt_states_inner_product_wrt_states(
            d_odes=lift_wrt_odes_gradient, d_states=states_wrt_mach,
        ), states_wrt_aoa,
    ) +
    solver.compute_odes_wrt_vertices_inner_product_wrt_mach(
        d_odes=lift_wrt_odes_gradient, d_vertices=vertices_wrt_aoa,
    ).item() +
    np.vdot(
        solver.compute_odes_wrt_states_inner_product_wrt_vertices(
            d_odes=lift_wrt_odes_gradient, d_states=states_wrt_mach,
        ), vertices_wrt_aoa,
    )
)

# Compute lift coefficient wrt pitch angle
coef_wrt_aoa = np.vdot(
    lift_wrt_forces_adj,
    solver.apply_forces_wrt_vertices_fwd(d_vertices=vertices_wrt_aoa) +
    solver.apply_forces_wrt_states_fwd(d_states=states_wrt_aoa),
) / (dynamic_pressure * args.chord)

# Compute second derivative of lift coefficient wrt pitch angle and Mach number
coef_wrt_aoa_wrt_mach = (
    lift_wrt_aoa_wrt_mach / (dynamic_pressure * args.chord) -
    2 / solver.mach_number * coef_wrt_aoa
)

# Print derivatives
print("\nLift coefficient derivatives:")
print(f"{'wrt pitch angle:':<36} {coef_wrt_aoa.real:.4f} (1/rad)")
print(f"{'wrt pitch angle wrt Mach number:':<36} {coef_wrt_aoa_wrt_mach.real:.4f} (1/rad)")
