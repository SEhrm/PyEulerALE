"""Wrapper class for the FORTRAN module

This module provides the Python class wrapping the FORTRAN subroutines implementing the spatial
discretization.

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

from pathlib import Path

import numpy as np
from scipy.sparse import bsr_array
from scipy.sparse.linalg import spsolve

from .euler_ale import spatial_discretization as disc

NUM_DIM = 2
NUM_VAR = 4
HEAT_RATIO = 1.4


class SpatialDiscretization:
    """Spatial discretization of the compressible 2D Euler equations around a moving airfoil

    Wraps FORTRAN subroutines implementing a central-scheme finite-volume discretization with
    Rusanov/Lax-Friedrich flux for a moving grid. The implementation provides linearization such
    that one can run both the ordinary differential equation `d𝓤/dt = 𝓡(𝓤,𝓧,𝓥), 𝐟 = 𝐟(𝓤,𝓧)` and
    the time-invariant state-space representation `dδ𝓤/dt = ∂𝓡/∂𝓤⋅δ𝓤 + ∂𝓡/∂𝓧⋅δ𝓧 + ∂𝓡/∂𝓥⋅dδ𝓧/dt,
    δ𝐟 = ∂𝐟/∂𝓤⋅δ𝓤 + ∂𝐟/∂𝓧⋅δ𝓧` for the cell averaged states `𝓤`, the grid vertices `𝓧`, the grid
    velocities `𝓥` and the total section force `𝐟`.

    The class allocates and holds the data arrays to be manipulated in-place by the FORTRAN
    subroutines. The general procedure for the user is to in-place modify the instance attribute
    arrays, call an instance method to process the array, and to read the updated array again.
    """

    _num_radial: int
    _num_angular: int
    _vertices: np.ndarray
    _states: np.ndarray
    _odes: np.ndarray
    _total_force: np.ndarray
    _surface_pressure_coefficients: np.ndarray
    # Jacobians:
    _odes_wrt_states: np.ndarray
    _odes_wrt_vertices: np.ndarray
    _odes_wrt_velocities: np.ndarray
    _total_force_wrt_states: np.ndarray
    _total_force_wrt_vertices: np.ndarray

    def __init__(
        self,
        grid_file: str | Path,
        angle_of_attack: float = 1.25,
        mach_numer: float = 0.5,
        rusanov_factor: float = 1e-1,
    ) -> None:
        """Initialize the discretization

        Args:
            grid_file: Grid file in PLOT3D format with single block. Only two-dimensional,
                structured, closed, O-type grids can be processed.
            angle_of_attack: Far-field angle of attack in degree.
            mach_numer: Mach number. The scheme is unlikely to produce reliable results for
                shocks in the sonic regine.
            rusanov_factor: Rusanov/Lax-Friedrich flux factor, typically between 0 and 1. Higher
                values increase stability and avoid oscillations in the solution but introduce but
                numerical dissipation decreasing the accuracy
        """
        self._vertices = self._read_grid(grid_file)
        self._num_radial = self._vertices.shape[1] - 1
        self._num_angular = self._vertices.shape[2] - 1
        self._surface_nodes \
            = np.mean([self._vertices[:, 0, :-1], self._vertices[:, 0, 1:]], axis=0)
        self._velocities = np.zeros_like(self._vertices)
        self._states = np.asfortranarray(np.empty(
            (NUM_VAR, self.num_radial, self.num_angular), dtype=complex,
        ))
        self._odes = np.asfortranarray(np.empty(
            (NUM_VAR, self.num_radial, self.num_angular), dtype=complex,
        ))
        self._total_force = np.asfortranarray(np.empty(
            2, dtype=complex,
        ))
        self._surface_pressure_coefficients = np.asfortranarray(np.empty(
            self.num_angular, dtype=float,
        ))
        self._odes_wrt_states = np.asfortranarray(np.zeros(
            (NUM_VAR, NUM_VAR, self.num_radial, self.num_angular, 5), dtype=float,
        ))
        self._odes_wrt_vertices = np.asfortranarray(np.zeros(
            (NUM_VAR, NUM_DIM, self.num_radial, self.num_angular, 4), dtype=float),
        )
        self._odes_wrt_velocities = np.asfortranarray(np.zeros(
            (NUM_VAR, NUM_DIM, self.num_radial, self.num_angular, 4), dtype=float),
        )
        self._total_force_wrt_states = np.asfortranarray(np.zeros(
            (NUM_DIM, NUM_VAR, self.num_angular), dtype=float),
        )
        self._total_force_wrt_vertices = np.asfortranarray(np.zeros(
            (NUM_DIM, NUM_DIM, self.num_angular, 2), dtype=float),
        )
        disc.aoa[...] = angle_of_attack
        disc.mach[...] = mach_numer
        disc.rusanov[...] = rusanov_factor
        disc.set_free_stream_state(self._states)

    @property
    def num_radial(self) -> int:
        """Number of cells in radial direction."""
        return self._num_radial

    @property
    def num_angular(self) -> int:
        """Number of cells in angular direction."""
        return self._num_angular

    @property
    def vertices(self) -> np.ndarray:
        """Grid vertices

        The current grid vertex coordinates in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.
        Initialized with the values read from the grid file.
        """
        return self._vertices.real

    @property
    def surface_points(self) -> np.ndarray:
        """Grid Surface points

        The current points of application for the surface pressure and section forces on the
        airfoil in shape ``(NUM_DIM,num_angular)``. Automatically recalculated from the current grid
        vertex coordinates as the midpoints between consecutive vertices on the airfoil surface.
        """
        return np.mean([self._vertices[:, 0, :-1].real, self._vertices[:, 0, 1:].real], axis=0)

    @property
    def velocities(self) -> np.ndarray:
        """Grid velocities

        The current grid vertex velocities in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.
        Initialized with zero.
        """
        return self._velocities.real

    @property
    def states(self) -> np.ndarray:
        """States

        The current cell averaged conserved variables in shape ``(NUM_VAR,num_radial,num_angular)``.
        Initialized as the free-stream.
        """
        return self._states.real

    @property
    def odes(self) -> np.ndarray:
        """States' rate of change (ODE)

        The current rate of change of the states in shape ``(NUM_VAR,num_radial,num_angular)``, i.e.
        the ordinary differential equation. Can be computed through ``compute_odes``.
        """
        return self._odes.real

    @property
    def total_force(self) -> np.ndarray:
        """Total section force

        The current total section force of the airfoil in shape ``(NUM_DIM,)``. Can be computed
        through ``compute_total_force``.
        """
        return self._total_force.real

    @property
    def surface_pressure_coefficients(self) -> np.ndarray:
        """Pressure coefficients on the airfoil surface

        The current pressure coefficients on the ``surface_points`` in shape ``(num_angular,)``.
        Can be computed through `compute_surface_pressure_coefficients``
        """
        return self._surface_pressure_coefficients.real

    @staticmethod
    def _read_grid(grid_file: str | Path) -> np.ndarray:
        """Reads the grid file

        Args:
            grid_file: Grid file in PLOT3D format with single block.

        Returns:
            Vertex points.
        """
        num_radial_pts, num_angular_pts = np.loadtxt(grid_file, dtype=int, skiprows=1, max_rows=1)
        vertex_points = np.asfortranarray(np.loadtxt(
            grid_file, dtype=float, skiprows=2,
        ).reshape((NUM_DIM, num_radial_pts, num_angular_pts)), dtype=complex)
        return vertex_points

    @staticmethod
    def _check_array(array: np.ndarray, shape: tuple, dtype: type = np.complex128) -> None:
        """Checks if the array is suitable to be passed to FORTRAN

        To avoid copies when passing to FORTRAN, an array must have the correct shape, data type,
        and must be FORTRAN-contiguous.

        Args:
            array: Array to be checked.
            shape: Required shape.
            dtype: Required data type.

        Raises:
             RuntimeError: If the array is unsuitable to be passed to FORTRAN.
        """
        if array.shape != shape:
            msg = f"Incorrect shape. Got {array.shape}, expected {shape}"
        elif array.dtype != dtype:
            msg = f"Incorrect data type. Got {array.dtype}, expected {dtype}"
        elif not array.flags.f_contiguous:
            msg = "Array is not FORTRAN-contiguous"
        else:
            return
        raise RuntimeError(msg)

    def compute_odes(self) -> None:
        """Computes the states' rate of change (ODE)

        Computes ``odes`` from ``states``, ``vertices`` and ``velocities``.
        """
        disc.compute_odes(self._vertices, self._velocities, self._states, self._odes)

    def linearize(self) -> None:
        """Computes the Jacobians"""
        disc.compute_odes_wrt_states(
            self._vertices, self._velocities, self._states, self._odes_wrt_states)
        disc.compute_odes_wrt_vertices(
            self._vertices, self._velocities, self._states, self._odes_wrt_vertices)
        disc.compute_odes_wrt_velocities(
            self._vertices, self._velocities, self._states, self._odes_wrt_velocities)
        disc.compute_total_force_wrt_states(
            self._vertices, self._states, self._total_force_wrt_states)
        disc.compute_total_force_wrt_vertices(
            self._vertices, self._states, self._total_force_wrt_vertices)

    def apply_odes_wrt_states_fwd(
        self,
        d_states: np.ndarray,
        d_odes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``odes`` with respect to ``states`` in forward mode

        Computes the matrix-vector-product `∂𝓡/∂𝓤⋅δ𝓤`, i.e. the directional derivative.

        Args:
            d_states: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_odes: Vector into which to store the vector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_VAR,num_radial,num_angular)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Vector-product.
        """
        self._check_array(d_states, self._states.shape)
        if d_odes is not None:
            self._check_array(d_odes, self._odes.shape)
        else:
            d_odes = np.empty_like(self._odes)
        disc.apply_odes_wrt_states_fwd(self._odes_wrt_states, d_states, d_odes)
        return d_odes

    def apply_odes_wrt_states_rev(
        self,
        d_odes: np.ndarray,
        d_states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``odes`` with respect to ``states`` in reverse mode

        Computes the matrix-vector-product `∂𝓡/∂𝓤ᵀ⋅δ𝓡`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be a complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_states: Covector into which to store the covector-product. Must be complex
                FORTRAN-contiguous array in shape ``(NUM_VAR,num_radial,num_angular)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Covector-product.
        """
        self._check_array(d_odes, self._odes.shape)
        if d_states is not None:
            self._check_array(d_states, self._states.shape)
        else:
            d_states = np.empty_like(self._states)
        disc.apply_odes_wrt_states_rev(self._odes_wrt_states, d_odes, d_states)
        return d_states

    def apply_odes_wrt_vertices_fwd(
        self,
        d_vertices: np.ndarray,
        d_odes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``odes`` with respect to ``vertices`` in forward mode

        Computes the matrix-vector-product `∂𝓡/∂𝓧⋅δ𝓧`, i.e. the directional derivative.

        Args:
            d_vertices: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.
            d_odes: Vector into which to store the vector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_VAR,num_radial,num_angular)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Vector-product.
        """
        self._check_array(d_vertices, self._vertices.shape)
        if d_odes is not None:
            self._check_array(d_odes, self._odes.shape)
        else:
            d_odes = np.empty_like(self._odes)
        disc.apply_odes_wrt_vertices_fwd(self._odes_wrt_vertices, d_vertices, d_odes)
        return d_odes

    def apply_odes_wrt_vertices_rev(
        self,
        d_odes: np.ndarray,
        d_vertices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``odes`` with respect to ``vertices`` in reverse mode

        Computes the matrix-vector-product `∂𝓡/∂𝓧ᵀ⋅δ𝓡`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be a complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_vertices: Covector into which to store the covector-product. Must be complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Covector-product.
        """
        self._check_array(d_odes, self._odes.shape)
        if d_vertices is not None:
            self._check_array(d_vertices, self._vertices.shape)
        else:
            d_vertices = np.empty_like(self._vertices)
        disc.apply_odes_wrt_vertices_rev(self._odes_wrt_vertices, d_odes, d_vertices)
        return d_vertices

    def apply_odes_wrt_velocities_fwd(
        self,
        d_velocities: np.ndarray,
        d_odes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``odes`` with respect to ``velocities`` in forward mode

        Computes the matrix-vector-product `∂𝓡/∂𝓧⋅δ𝓥`, i.e. the directional derivative.

        Args:
            d_velocities: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.
            d_odes: Vector into which to store the vector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_VAR,num_radial,num_angular)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Vector-product.
        """
        self._check_array(d_velocities, self._velocities.shape)
        if d_odes is not None:
            self._check_array(d_odes, self._odes.shape)
        else:
            d_odes = np.empty_like(self._odes)
        # reusing same functionality
        disc.apply_odes_wrt_vertices_fwd(self._odes_wrt_velocities, d_velocities, d_odes)
        return d_odes

    def apply_odes_wrt_velocities_rev(
        self,
        d_odes: np.ndarray,
        d_velocities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``odes`` with respect to ``velocities`` in reverse mode

        Computes the matrix-vector-product `∂𝓡/∂𝓧ᵀ⋅δ𝓡`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be a complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_velocities: Covector into which to store the covector-product. Must be complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Covector-product.
        """
        self._check_array(d_odes, self._odes.shape)
        if d_velocities is not None:
            self._check_array(d_velocities, self._velocities.shape)
        else:
            d_velocities = np.empty_like(self._velocities)
        # reusing same functionality
        disc.apply_odes_wrt_vertices_rev(self._odes_wrt_velocities, d_odes, d_velocities)
        return d_velocities

    def solve_odes_wrt_states_fwd(
        self,
        d_odes: np.ndarray,
        shift: float | complex = 0.,
        d_states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solves Jacobians of ``odes`` with respect to ``states`` in forward mode

        Computes the shifted inverse-vector-product `(∂𝓡/∂𝓤 - σ⋅Id)⁻¹⋅δ𝓡`.

        Args:
            d_odes: Vector to multiply to the Inverse. Must be a (complex) FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            shift: Shift.
            d_states: Vector into which to store the Vector-product. Must be (complex)
                FORTRAN-contiguous array in shape ``(NUM_VAR,num_radial,num_angular)``. if not
                provided, a newly-allocated (complex) array will be returned.

        Returns:
            Vector-product.
        """
        result_dtype = complex if (np.iscomplex(shift) or d_odes.dtype == complex) else float
        if d_states is None:
            d_states = np.empty_like(d_odes, result_dtype)
        else:
            self._check_array(d_states, self._states.shape, result_dtype)
        # get jacobians as real sparse CSR matrix
        indices = np.empty((self.num_radial * 5 - 2) * self.num_angular, dtype=np.intc)
        data = np.empty((4, 4, len(indices)), dtype=float, order="F")
        index_pointers = np.empty(self.num_angular * self.num_radial + 1, dtype=np.intc)
        disc.convert_odes_wrt_states(self._odes_wrt_states, data, indices, index_pointers)
        jacobi = bsr_array((
            np.moveaxis(data, -1, 0), indices - 1, index_pointers - 1,
        ), dtype=type(shift)).tocsr()
        # apply possibly complex shift
        if shift != 0.:
            jacobi.setdiag(jacobi.diagonal() - shift)
        # sparse LU for possibly complex solution
        d_states.ravel(order="K")[:] = spsolve(
            jacobi, d_odes.ravel(order="K"),
        )
        return d_states

    def compute_total_force(self) -> None:
        """Computes the total section force

        Computes ``total_force`` from ``states`` and ``vertices``.

        """
        disc.compute_total_force(self._vertices, self._states, self._total_force)

    def apply_total_force_wrt_states_fwd(
        self,
        d_states: np.ndarray,
        d_total_force: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``total_force`` with respect to ``states`` in forward mode

        Computes the matrix-vector-product `∂𝐟/∂𝓤⋅δ𝓤`, i.e. the directional derivative.

        Args:
            d_states: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_total_force: Vector into which to store the vector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,)``. if not provided, a newly-allocated
                array will be returned.

        Returns:
            Vector-product.
        """
        self._check_array(d_states, self._states.shape)
        if d_total_force is not None:
            self._check_array(d_total_force, self._total_force.shape)
        else:
            d_total_force = np.empty_like(self._total_force)
        disc.apply_total_force_wrt_states_fwd(
            self._total_force_wrt_states, d_states, d_total_force)
        return d_total_force

    def apply_total_force_wrt_states_rev(
        self,
        d_total_force: np.ndarray,
        d_states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``total_force`` with respect to ``states`` in reverse mode

        Computes the matrix-vector-product `∂𝐟/∂𝓤ᵀ⋅δ𝐟`.

        Args:
            d_total_force: Covector to multiply to the Jacobians. Must be complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,)``.
            d_states: Covector into which to store the covector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,num_radial,num_angular)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Covector-product.
        """
        self._check_array(d_total_force, self._total_force.shape)
        if d_states is not None:
            self._check_array(d_states, self._states.shape)
        else:
            d_states = np.empty_like(self._states)
        self._check_array(d_total_force, self._total_force.shape)
        disc.apply_total_force_wrt_states_rev(
            self._total_force_wrt_states, d_total_force, d_states)
        return d_states

    def apply_total_force_wrt_vertices_fwd(
        self,
        d_vertices: np.ndarray,
        d_total_force: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``total_force`` with respect to ``vertices`` in forward mode

        Computes the matrix-vector-product `∂𝐟/∂𝓧⋅δ𝓧`, i.e. the directional derivative.

        Args:
            d_vertices: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.
            d_total_force: Vector into which to store the vector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,)``. if not provided, a newly-allocated
                array will be returned.

        Returns:
            Vector-product.
        """
        self._check_array(d_vertices, self._vertices.shape)
        if d_total_force is not None:
            self._check_array(d_total_force, self._total_force.shape)
        else:
            d_total_force = np.empty_like(self._total_force)
        disc.apply_total_force_wrt_vertices_fwd(
            self._total_force_wrt_vertices, d_vertices, d_total_force)
        return d_total_force

    def apply_total_force_wrt_vertices_rev(
        self,
        d_total_force: np.ndarray,
        d_vertices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``total_force`` with respect to ``vertices`` in reverse mode

        Computes the matrix-vector-product `∂𝐟/∂𝓧ᵀ⋅δ𝐟`.

        Args:
            d_total_force: Covector to multiply to the Jacobians. Must be complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,)``.
            d_vertices: Covector into which to store the covector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Covector-product.
        """
        self._check_array(d_total_force, self._total_force.shape)
        if d_vertices is not None:
            self._check_array(d_vertices, self._vertices.shape)
        else:
            d_vertices = np.empty_like(self._vertices)
        disc.apply_total_force_wrt_vertices_rev(
            self._total_force_wrt_vertices, d_total_force, d_vertices)
        return d_vertices

    def compute_surface_pressure_coefficients(self) -> None:
        """Computes the pressure coefficients on the airfoil surface

        Computes ``surface_pressure_coefficients`` from ``states``.
        """
        disc.compute_pressure_coeffs(self._states, self._surface_pressure_coefficients)
