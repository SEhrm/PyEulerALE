"""Finite volume discretization for the 2D compressible Euler equations around moving airfoils.

This package implements a finite volume method for the spatial discretization of the two-dimensional
compressible Euler equations around moving airfoils in arbitrary Lagrangian-Eulerian formulation
(ALE). The discretization uses a central-scheme with artificial dissipation for structured
O-type grids.

Key features:
    * Moving grid capabilities with arbitrary deformation (ALE formulation)
    * Complex-valued spatial discretization procedures implemented in FORTRAN
    * Data management and program flow in Python
    * Computation of all Jacobians by complex-step for linearized state-space formulation
    * Forward and reverse (adjoint) application of the Jacobians for sensitivity analysis
    * Sparse matrix solver for the resolvent of the linear system

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

from .core import HEAT_RATIO
from .core import NUM_DIM
from .core import NUM_VAR
from .core import SpatialDiscretization
from .utils import get_pressure_coefficient
from .utils import get_pressure_coefficient_derivative
from .utils import get_section_coefficients

__all__ = [
    "HEAT_RATIO",
    "NUM_DIM",
    "NUM_VAR",
    "SpatialDiscretization",
    "get_pressure_coefficient",
    "get_pressure_coefficient_derivative",
    "get_section_coefficients",
]
