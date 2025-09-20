#!/usr/bin/env python

"""Solves the compressible Euler equations around an airfoil for steady-state condition

Pseudo-transient continuation (PTC) is used to iterate the nonlinear ordinary differential
equation in pseudo-time by linearized backward-Euler. The global pseudo time-step size is controlled
by switched evolution relaxation (SER) where the time-step size is inverse proportional to the
residual norm.

The script prints the coefficient of lift, and saves the pressure coefficients on the surface of
the airfoil to ``cp.csv``. By linear airfoil theory, the resulting the steady-state coefficient of
lift should be `cₗ = 2π / √(1−Ma₀₀²) ⋅ (α/1ʳ + ḣ / ‖𝐯₀₀‖)` where `α` is the far-field
angle-of-attack and `ḣ / ‖𝐯₀₀‖` is the (downward) plunge speed of the airfoil as fraction of the
free-stream speed.

Example:
    python run.py ../meshes/65x65.x 0.5 1.25 0.0 1.0

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

from argparse import ArgumentParser
from pathlib import Path
from timeit import default_timer as timer

import numpy as np

from py_euler_ale import HEAT_RATIO, SpatialDiscretization

parser = ArgumentParser(
    description="Solves the compressible Euler equations around an airfoil for steady-state "
                "condition at constant far-field angle-of-attack and plunge speed.")
parser.add_argument("mesh_file", type=str, help="Mesh file.")
parser.add_argument("mach_number", type=float, help="Free-stream Mach number.")
parser.add_argument("angle_of_attack", type=float, help="Far-field angle-of-attack (in deg).")
parser.add_argument("plunge_speed", type=float, help="Plunge speed (in free-stream speed).")
parser.add_argument("chord", type=float, help="Chord length (in grid units).")
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

print(f"{'Mesh file:':<25} {args.mesh_file}")
print(f"{'Mach number:':<25} {args.mach_number}")
print(f"{'Angle of attack (deg):':<25} {args.angle_of_attack}")
print(f"{'Plunge speed / |vₒₒ|:':<25} {args.plunge_speed}")
print(f"{'Chord / L:':<25} {args.chord}")
print(f"{'Rusanov flux factor:':<25} {args.rusanov}")

# Non-dimensional free-stream speed `v₀₀/√(p₀₀/ϱ₀₀) = Ma₀₀⋅√γ`
free_stream_speed = args.mach_number * HEAT_RATIO**0.5

# Initialize solver
solver = SpatialDiscretization(
    grid_file=args.mesh_file,
    mach_numer=args.mach_number,
    angle_of_attack=args.angle_of_attack,
    rusanov_factor=args.rusanov,
)

# Set the non-dimensional grid velocities ``solver.velocities`` in the z-direction based on the
# plunge speed, i.e. `v = -ḣ/√(p₀₀/ϱ₀₀) = -(ḣ/v₀₀)⋅v₀₀/√(p₀₀/ϱ₀₀) = -(ḣ/v₀₀)⋅(Ma₀₀⋅√γ)`
solver.velocities[1, :] = -args.plunge_speed * free_stream_speed

# Compute ``solver.odes`` based on ``solver.states`` which is initialized by free-stream
solver.compute_odes()

# Compute the ∞-norm of ``solver.odes`` associated to the free-stream state
norm_free_stream = np.max(np.abs(solver.odes))

# Save start time
time = timer()

# Run pseudo-transient continuation (PTC)
print(f"\n{'it':>6} {'residual':>15} {'rel residual':>15} {'coef lift':>15}")
for nt in range(args.iter):

    # Compute current ``solver.odes`` based on current ``solver.states``
    solver.compute_odes()

    # Compute the ∞-norm of ``solver.odes``
    norm = np.max(np.abs(solver.odes))

    # Compute relative norm
    rel_norm = norm / norm_free_stream

    # Compute current ``solver.total_force`` based current ``solver.states``
    solver.compute_total_force()

    # Compute the classical lift coefficient from the non-dimensional total force
    _, lift_coef = solver.total_force / (args.mach_number**2 * HEAT_RATIO / 2 * args.chord)

    # Print the current iterate
    print(f"{nt:>6} {norm:>15.3e} {rel_norm:>15.3e} {lift_coef:>+15.3e}")

    # If satisfied, break
    if rel_norm <= args.rtol or np.isnan(rel_norm):
        break

    # Linearize the solver based on the current ``solver.states``
    solver.linearize()

    # Get the global pseudo time-step size by switched evolution relaxation (SER)
    time_step_size = 1. * rel_norm**-1.

    # Solve linearized system for the update based on the time-step size and the current residual
    update = solver.solve_odes_wrt_states_fwd(
        d_odes=-solver.odes, shift=time_step_size**-1.,
    )

    # Update ``solver.states``
    solver.states[:] += update

# Report elapsed time
print(f"Elapsed time: {timer() - time:.2f} sec")

# Compute ``solver.pressure_coefficients`` based on ``solver.states``
solver.compute_surface_pressure_coefficients()

# Export ``solver.pressure_coefficients`` and ``solver.surface_points`` for postprocessing
path = Path(__file__).parent / Path("cp.csv")
np.savetxt(
    fname=path, fmt="%.5e", delimiter=",", header="x,y,cp",
    X=np.vstack((solver.surface_points / args.chord, solver.surface_pressure_coefficients)).T,
)
print(f"\nSaved surface pressure coefficients to 'cp.csv' ('{path}')")
