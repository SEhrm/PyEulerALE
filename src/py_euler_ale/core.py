"""Wrapper class for the FORTRAN module

This module provides the Python class wrapping the FORTRAN subroutines implementing the spatial
discretization.

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import bsr_array
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve

from .euler_ale import spatial_discretization as disc

NUM_DIM = 2
NUM_VAR = 4
HEAT_RATIO = 1.4


class SpatialDiscretization:
    """Spatial discretization of the compressible 2D Euler equations around a moving airfoil

    Wraps FORTRAN subroutines implementing a central-scheme finite-volume discretization with
    artificial dissipation for a moving grid. The implementation provides linearization such
    that one can run both the ordinary differential equation `d𝓤/dt = 𝓡(𝓤,𝓧,𝓥), 𝓕 = 𝓕(𝓤,𝓧)` and
    the time-invariant state-space representation `dδ𝓤/dt = ∂𝓡/∂𝓤⋅δ𝓤 + ∂𝓡/∂𝓧⋅δ𝓧 + ∂𝓡/∂𝓥⋅dδ𝓧/dt,
    δ𝓕 = ∂𝓕/∂𝓤⋅δ𝓤 + ∂𝓕/∂𝓧⋅δ𝓧` for the cell averaged states `𝓤`, the grid vertices `𝓧`, the grid
    velocities `𝓥` and the forces `𝓕`.

    The class allocates and holds the data arrays to be manipulated in-place by the FORTRAN
    subroutines. The general procedure for the user is to in-place modify the instance attribute
    arrays, call an instance method to process the array, and to read the updated array again.
    """

    _k2: float
    _k4: float
    _c2: float
    _mach: complex
    _aoa: float
    _num_radial: int
    _num_angular: int
    _vertices: np.ndarray
    _states: np.ndarray
    _odes: np.ndarray
    _forces: np.ndarray
    # Jacobians:
    _odes_wrt_mach: np.ndarray
    _odes_wrt_states: np.ndarray
    _odes_wrt_vertices: np.ndarray
    _odes_wrt_velocities: np.ndarray
    _forces_wrt_states: np.ndarray
    _forces_wrt_vertices: np.ndarray

    def __init__(
        self,
        grid_file: str | Path,
        angle_of_attack: float = 1.25,
        mach_number: float = 0.5,
        jst_k2: float = 1,
        jst_k4: float = 1 / 32,
        jst_c4: float = 2,
    ) -> None:
        """Initialize the discretization

        Args:
            grid_file: Grid file in PLOT3D format with single block. Only two-dimensional,
                structured, closed, O-type grids can be processed.
            angle_of_attack: Far-field angle of attack in degree.
            mach_number: Free-stream Mach number. The scheme is unlikely to produce reliable results
                for shocks in the sonic regine.
            jst_k2: JST artificial dissipation constant `κ₂`
            jst_k4: JST artificial dissipation constant `κ₄`
            jst_c4: JST artificial dissipation constant `c₄`
        """
        self.mach_number = mach_number
        self.angle_of_attack = angle_of_attack
        self._k2, self._k4, self._c4 = jst_k2, jst_k4, jst_c4
        self._vertices = self._read_grid(grid_file)
        self._num_radial = self._vertices.shape[1] - 1
        self._num_angular = self._vertices.shape[2] - 1
        self._velocities = np.zeros_like(self._vertices)
        self._states = np.asfortranarray(np.empty(dtype=complex, shape=(
            NUM_VAR, self.num_radial, self.num_angular)))
        self._odes = np.asfortranarray(np.empty(dtype=complex, shape=(
            NUM_VAR, self.num_radial, self.num_angular)))
        self._forces = np.asfortranarray(np.empty(dtype=complex, shape=(
            NUM_DIM, self.num_angular)))
        self._odes_wrt_mach = np.asfortranarray(np.zeros(dtype=float, shape=(
            NUM_VAR, 1, self.num_radial, self.num_angular)))
        self._odes_wrt_states = np.asfortranarray(np.zeros(dtype=float, shape=(
            NUM_VAR, NUM_VAR, self.num_radial, self.num_angular, 9)))
        self._odes_wrt_vertices = np.asfortranarray(np.zeros(dtype=float, shape=(
            NUM_VAR, NUM_DIM, self.num_radial, self.num_angular, 4)))
        self._odes_wrt_velocities = np.asfortranarray(np.zeros(dtype=float, shape=(
            NUM_VAR, NUM_DIM, self.num_radial, self.num_angular, 4)))
        self._forces_wrt_states = np.asfortranarray(np.zeros(dtype=float, shape=(
            NUM_DIM, NUM_VAR, self.num_angular)))
        self._forces_wrt_vertices = np.asfortranarray(np.zeros(dtype=float, shape=(
            NUM_DIM, NUM_DIM, self.num_angular, 2)))
        self.set_free_stream_state()

    @property
    def mach_number(self) -> float:
        """Free-stream Mach number"""
        return self._mach.real

    @mach_number.setter
    def mach_number(self, mach_number: float) -> None:
        self._mach = complex(mach_number)

    @property
    def angle_of_attack(self) -> float:
        """Far-field angle of attack in degree"""
        return self._aoa

    @angle_of_attack.setter
    def angle_of_attack(self, angle_of_attack: float) -> None:
        self._aoa = angle_of_attack

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
    def forces(self) -> np.ndarray:
        """Section forces

        The current section forces on the ``surface_points`` in shape ``(NUM_DIM,num_angular)``.
        Can be computed through ``compute_forces``.
        """
        return self._forces.real

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
    def _check_array(
        array: np.ndarray,
        shape: tuple,
        dtype: np.dtype = np.dtypes.Complex128DType(),
    ) -> None:
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
            msg = f"Incorrect data type. Got {array.dtype.name}, expected {np.dtype(dtype).name}"
        elif not array.flags.f_contiguous:
            msg = "Array is not FORTRAN-contiguous"
        else:
            return
        raise RuntimeError(msg)

    def _configure_disc(self) -> None:
        """Configures the discretization with current parameters"""
        disc.aoa[...] = self._aoa
        disc.jst_k2[...] = self._k2
        disc.jst_k4[...] = self._k4
        disc.jst_c4[...] = self._c4

    def set_free_stream_state(self) -> None:
        """Sets ``states`` to free-stream"""
        self._configure_disc()
        disc.set_free_stream_state(self.mach_number, self._states)

    def compute_odes(self) -> None:
        """Computes the states' rate of change (ODE)

        Computes ``odes`` from ``states``, ``vertices`` and ``velocities``.
        """
        self._configure_disc()
        disc.compute_odes(self._mach, self._vertices, self._velocities, self._states, self._odes)

    def linearize(self) -> None:
        """Computes the Jacobians"""
        self._configure_disc()
        disc.compute_odes_wrt_mach(
            self._mach, self._vertices, self._velocities, self._states, self._odes_wrt_mach)
        disc.compute_odes_wrt_states(
            self._mach, self._vertices, self._velocities, self._states, self._odes_wrt_states)
        disc.compute_odes_wrt_vertices(
            self._mach, self._vertices, self._velocities, self._states, self._odes_wrt_vertices)
        disc.compute_odes_wrt_velocities(
            self._mach, self._vertices, self._velocities, self._states, self._odes_wrt_velocities)
        disc.compute_forces_wrt_states(
            self._vertices, self._states, self._forces_wrt_states)
        disc.compute_forces_wrt_vertices(
            self._vertices, self._states, self._forces_wrt_vertices)

    def apply_odes_wrt_mach_fwd(
        self,
        d_mach: np.ndarray,
        d_odes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``odes`` with respect to ``mach_number`` in forward mode

        Computes the matrix-vector-product `∂𝓡/∂Maₒₒ⋅δMaₒₒ`, i.e. the directional derivative.

        Args:
            d_mach: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(1)``.
            d_odes: Vector into which to store the vector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_VAR,num_radial,num_angular)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Vector-product.
        """
        self._check_array(d_mach, (1,))
        if d_odes is not None:
            self._check_array(d_odes, self._odes.shape)
        else:
            d_odes = np.empty_like(self._odes)
        disc.apply_odes_wrt_mach_fwd(self._odes_wrt_mach, d_mach, d_odes)
        return d_odes

    def apply_odes_wrt_mach_rev(
        self,
        d_odes: np.ndarray,
        d_mach: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``odes`` with respect to ``mach_number`` in reverse mode

        Computes the matrix-vector-product `∂𝓡/∂Maₒₒᵀ⋅δ𝓡`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be a complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_mach: Covector into which to store the covector-product. Must be complex
                FORTRAN-contiguous array in shape ``(1)``. if not provided, a newly-allocated array
                will be returned.

        Returns:
            Covector-product.
        """
        self._check_array(d_odes, self._odes.shape)
        if d_mach is not None:
            self._check_array(d_mach, (1,))
        else:
            d_mach = np.empty((1,), dtype=complex)
        disc.apply_odes_wrt_mach_rev(self._odes_wrt_mach, d_odes, d_mach)
        return d_mach

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

    def _assemble_csr(self, shift: float | complex = 0.) -> csr_array:
        """Assembles the Jacobians of ``odes`` with respect to ``states`` as sparse CSR matrix.

        Args:
            shift: Shift value to subtract from the main diagonal.

        Returns:
            (Shifted) Jacobian as CSR matrix.
        """
        indices = np.empty((self.num_radial * 9 - 6) * self.num_angular, dtype=np.intc)
        data = np.empty((4, 4, len(indices)), dtype=float, order="F")
        index_pointers = np.empty(self.num_angular * self.num_radial + 1, dtype=np.intc)
        disc.convert_odes_wrt_states(self._odes_wrt_states, data, indices, index_pointers)
        jacobi = bsr_array((
            np.moveaxis(data, -1, 0), indices - 1, index_pointers - 1,
        ), dtype=type(shift)).tocsr()
        # apply possibly complex shift
        if shift != 0.:
            jacobi.setdiag(jacobi.diagonal() - shift)
        return jacobi

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
        self._check_array(d_odes, self._odes.shape, dtype=d_odes.dtype)  # ignore dtype
        result_dtype = np.promote_types(type(shift), d_odes.dtype)
        if d_states is None:
            d_states = np.empty_like(d_odes, result_dtype)
        else:
            self._check_array(d_states, self._states.shape, result_dtype)
        jacobi = self._assemble_csr(shift=shift)
        # sparse LU for possibly complex solution
        d_states.ravel(order="K")[:] = spsolve(
            jacobi, d_odes.ravel(order="K"),
        )
        return d_states

    def solve_odes_wrt_states_adj(
        self,
        d_states: np.ndarray,
        shift: float | complex = 0.,
        d_odes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solves Jacobians of ``odes`` with respect to ``states`` in reverse mode

        Computes the shifted inverse-vector-product `(∂𝓡/∂𝓤 - σ⋅Id)⁻ᴴ⋅δ𝓤`.

        Args:
            d_states: Covector to multiply to the Inverse. Must be a (complex) FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            shift: Shift.
            d_odes: Covector into which to store the covector-product. Must be (complex)
                FORTRAN-contiguous array in shape ``(NUM_VAR,num_radial,num_angular)``. if not
                provided, a newly-allocated (complex) array will be returned.

        Returns:
            Covector-product.
        """
        self._check_array(d_states, self._odes.shape, dtype=d_states.dtype)  # ignore dtype
        result_dtype = np.promote_types(type(shift), d_states.dtype)
        if d_odes is None:
            d_odes = np.empty_like(d_states, result_dtype)
        else:
            self._check_array(d_odes, self._odes.shape, result_dtype)
        jacobi = self._assemble_csr(shift=shift.conjugate())
        # sparse LU for possibly complex solution
        d_odes.ravel(order="K")[:] = spsolve(
            jacobi.T, d_states.ravel(order="K"),
        )
        return d_odes

    def compute_forces(self) -> None:
        """Computes the section forces

        Computes ``forces`` from ``states`` and ``vertices``.

        """
        self._configure_disc()
        disc.compute_forces(self._vertices, self._states, self._forces)

    def apply_forces_wrt_states_fwd(
        self,
        d_states: np.ndarray,
        d_forces: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``forces`` with respect to ``states`` in forward mode

        Computes the matrix-vector-product `∂𝓕/∂𝓤⋅δ𝓤`, i.e. the directional derivative.

        Args:
            d_states: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_forces: Vector into which to store the vector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,num_angular)``. if not provided, a
                newly-allocated array will be returned.

        Returns:
            Vector-product.
        """
        self._check_array(d_states, self._states.shape)
        if d_forces is not None:
            self._check_array(d_forces, self._forces.shape)
        else:
            d_forces = np.empty_like(self._forces)
        disc.apply_forces_wrt_states_fwd(self._forces_wrt_states, d_states, d_forces)
        return d_forces

    def apply_forces_wrt_states_rev(
        self,
        d_forces: np.ndarray,
        d_states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``forces`` with respect to ``states`` in reverse mode

        Computes the matrix-vector-product `∂𝓕/∂𝓤ᵀ⋅δ𝓕`.

        Args:
            d_forces: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_angular)``.
            d_states: Covector into which to store the covector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_VAR,num_radial,num_angular)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Covector-product.
        """
        self._check_array(d_forces, self._forces.shape)
        if d_states is not None:
            self._check_array(d_states, self._states.shape)
        else:
            d_states = np.empty_like(self._states)
        disc.apply_forces_wrt_states_rev(self._forces_wrt_states, d_forces, d_states)
        return d_states

    def apply_forces_wrt_vertices_fwd(
        self,
        d_vertices: np.ndarray,
        d_forces: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``forces`` with respect to ``vertices`` in forward mode

        Computes the matrix-vector-product `∂𝓕/∂𝓧⋅δ𝓧`, i.e. the directional derivative.

        Args:
            d_vertices: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.
            d_forces: Vector into which to store the vector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,num_angular)``. if not provided, a
                newly-allocated array will be returned.

        Returns:
            Vector-product.
        """
        self._check_array(d_vertices, self._vertices.shape)
        if d_forces is not None:
            self._check_array(d_forces, self._forces.shape)
        else:
            d_forces = np.empty_like(self._forces)
        disc.apply_forces_wrt_vertices_fwd(self._forces_wrt_vertices, d_vertices, d_forces)
        return d_forces

    def apply_forces_wrt_vertices_rev(
        self,
        d_forces: np.ndarray,
        d_vertices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies Jacobians of ``forces`` with respect to ``vertices`` in reverse mode

        Computes the matrix-vector-product `∂𝓕/∂𝓧ᵀ⋅δ𝓕`.

        Args:
            d_forces: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_angular)``.
            d_vertices: Covector into which to store the covector-product. Must be a complex
                FORTRAN-contiguous array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``. if not
                provided, a newly-allocated array will be returned.

        Returns:
            Covector-product.
        """
        self._check_array(d_forces, self._forces.shape)
        if d_vertices is not None:
            self._check_array(d_vertices, self._vertices.shape)
        else:
            d_vertices = np.empty_like(self._vertices)
        disc.apply_forces_wrt_vertices_rev(self._forces_wrt_vertices, d_forces, d_vertices)
        return d_vertices

    def compute_odes_wrt_states_inner_product_wrt_mach(
        self,
        d_odes: np.ndarray,
        d_states: np.ndarray,
    ) -> np.ndarray:
        """Computes ``odes`` wrt ``states`` inner-product wrt ``mach_number``.

        Computes the gradient `∂<δ𝓡,∂𝓡/∂𝓤⋅δ𝓤>/∂Maₒₒ`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_states: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.

        Returns:
            Gradient in shape ``(1)``.
        """
        self._check_array(d_states, self._states.shape)
        self._check_array(d_odes, self._odes.shape)
        gradient = np.zeros((1,), dtype=complex)
        vec_jac_product = np.zeros((1,), dtype=complex)
        odes_wrt_mach_pert = np.zeros_like(self._odes_wrt_mach)
        step = 1e-6 * (1. + np.linalg.norm(self._states))**0.5 / np.linalg.norm(d_states)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_odes_wrt_mach(
                self._mach, self._vertices, self._velocities,
                self._states + sign * step * part(d_states), odes_wrt_mach_pert,
            )
            disc.apply_odes_wrt_mach_rev(
                odes_wrt_mach_pert, d_odes.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_odes_wrt_states_inner_product_wrt_states(
        self,
        d_odes: np.ndarray,
        d_states: np.ndarray,
    ) -> np.ndarray:
        """Computes ``odes`` wrt ``states`` inner-product wrt ``states``.

        Computes the gradient `∂<δ𝓡,∂𝓡/∂𝓤⋅δ𝓤>/∂𝓤`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_states: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.

        Returns:
            Gradient in shape `(NUM_VAR,num_radial,num_angular)``.
        """
        self._check_array(d_states, self._states.shape)
        self._check_array(d_odes, self._odes.shape)
        gradient = np.zeros_like(self._states)
        vec_jac_product = np.zeros_like(self._states)
        odes_wrt_states_pert = np.zeros_like(self._odes_wrt_states)
        step = 1e-6 * (1. + np.linalg.norm(self._states))**0.5 / np.linalg.norm(d_states)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_odes_wrt_states(
                self._mach, self._vertices, self._velocities,
                self._states + sign * step * part(d_states), odes_wrt_states_pert,
            )
            disc.apply_odes_wrt_states_rev(
                odes_wrt_states_pert, d_odes.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_odes_wrt_states_inner_product_wrt_vertices(
        self,
        d_odes: np.ndarray,
        d_states: np.ndarray,
    ) -> np.ndarray:
        """Computes ``odes`` wrt ``states`` inner-product wrt ``vertices``.

        Computes the gradient `∂<δ𝓡,∂𝓡/∂𝓤⋅δ𝓤>/∂𝓧`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_states: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.

        Returns:
            Gradient in shape `(NUM_DIM,num_radial+1,num_angular+1)``.
        """
        self._check_array(d_states, self._states.shape)
        self._check_array(d_odes, self._odes.shape)
        gradient = np.zeros_like(self._vertices)
        vec_jac_product = np.zeros_like(self._vertices)
        odes_wrt_vertices_pert = np.zeros_like(self._odes_wrt_vertices)
        step = 1e-6 * (1. + np.linalg.norm(self._states))**0.5 / np.linalg.norm(d_states)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_odes_wrt_vertices(
                self._mach, self._vertices, self._velocities,
                self._states + sign * step * part(d_states), odes_wrt_vertices_pert,
            )
            disc.apply_odes_wrt_vertices_rev(
                odes_wrt_vertices_pert, d_odes.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_odes_wrt_vertices_inner_product_wrt_mach(
        self,
        d_odes: np.ndarray,
        d_vertices: np.ndarray,
    ) -> np.ndarray:
        """Computes ``odes`` wrt ``vertices`` inner-product wrt ``mach``.

        Computes the gradient `∂<δ𝓡,∂𝓡/∂𝓧⋅δ𝓧>/∂Maₒₒ`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_vertices: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.

        Returns:
            Gradient in shape `(1)``.
        """
        self._check_array(d_vertices, self._vertices.shape)
        self._check_array(d_odes, self._odes.shape)
        gradient = np.zeros((1,), dtype=complex)
        vec_jac_product = np.zeros((1,), dtype=complex)
        odes_wrt_mach_pert = np.zeros_like(self._odes_wrt_mach)
        step = 1e-6 * (1. + np.linalg.norm(self._vertices))**0.5 / np.linalg.norm(d_vertices)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_odes_wrt_mach(
                self._mach, self._vertices + sign * step * part(d_vertices), self._velocities,
                self._states, odes_wrt_mach_pert,
            )
            disc.apply_odes_wrt_mach_rev(
                odes_wrt_mach_pert, d_odes.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_odes_wrt_vertices_inner_product_wrt_states(
        self,
        d_odes: np.ndarray,
        d_vertices: np.ndarray,
    ) -> np.ndarray:
        """Computes ``odes`` wrt ``vertices`` inner-product wrt ``states``.

        Computes the gradient `∂<δ𝓡,∂𝓡/∂𝓧⋅δ𝓧>/∂𝓤`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_vertices: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.

        Returns:
            Gradient in shape `(NUM_VAR,num_radial,num_angular)``.
        """
        self._check_array(d_vertices, self._vertices.shape)
        self._check_array(d_odes, self._odes.shape)
        gradient = np.zeros_like(self._states)
        vec_jac_product = np.zeros_like(self._states)
        odes_wrt_states_pert = np.zeros_like(self._odes_wrt_states)
        step = 1e-6 * (1. + np.linalg.norm(self._vertices))**0.5 / np.linalg.norm(d_vertices)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_odes_wrt_states(
                self._mach, self._vertices + sign * step * part(d_vertices), self._velocities,
                self._states, odes_wrt_states_pert,
            )
            disc.apply_odes_wrt_states_rev(
                odes_wrt_states_pert, d_odes.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_odes_wrt_vertices_inner_product_wrt_vertices(
        self,
        d_odes: np.ndarray,
        d_vertices: np.ndarray,
    ) -> np.ndarray:
        """Computes ``odes`` wrt ``vertices`` inner-product wrt ``vertices``.

        Computes the gradient `∂<δ𝓡,∂𝓡/∂𝓧⋅δ𝓧>/∂𝓧`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_vertices: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.

        Returns:
            Gradient in shape `(NUM_DIM,num_radial+1,num_angular+1)``.
        """
        self._check_array(d_vertices, self._vertices.shape)
        self._check_array(d_odes, self._odes.shape)
        gradient = np.zeros_like(self._vertices)
        vec_jac_product = np.zeros_like(self._vertices)
        odes_wrt_vertices_pert = np.zeros_like(self._odes_wrt_vertices)
        step = 1e-6 * (1. + np.linalg.norm(self._vertices))**0.5 / np.linalg.norm(d_vertices)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_odes_wrt_vertices(
                self._mach, self._vertices + sign * step * part(d_vertices), self._velocities,
                self._states, odes_wrt_vertices_pert,
            )
            disc.apply_odes_wrt_vertices_rev(
                odes_wrt_vertices_pert, d_odes.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_odes_wrt_velocities_inner_product_wrt_mach(
        self,
        d_odes: np.ndarray,
        d_velocities: np.ndarray,
    ) -> np.ndarray:
        """Computes ``odes`` wrt ``velocities`` inner-product wrt ``mach``.

        Computes the gradient `∂<δ𝓡,∂𝓡/∂𝓥⋅δ𝓥>/∂Maₒₒ`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_velocities: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.

        Returns:
            Gradient in shape `(1)``.
        """
        self._check_array(d_velocities, self._vertices.shape)
        self._check_array(d_odes, self._odes.shape)
        gradient = np.zeros((1,), dtype=complex)
        vec_jac_product = np.zeros((1,), dtype=complex)
        odes_wrt_mach_pert = np.zeros_like(self._odes_wrt_mach)
        step = 1e-6 * (1. + np.linalg.norm(self._velocities))**0.5 / np.linalg.norm(d_velocities)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_odes_wrt_mach(
                self._mach, self._vertices, self._velocities + sign * step * part(d_velocities),
                self._states, odes_wrt_mach_pert,
            )
            disc.apply_odes_wrt_mach_rev(
                odes_wrt_mach_pert, d_odes.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_odes_wrt_velocities_inner_product_wrt_states(
        self,
        d_odes: np.ndarray,
        d_velocities: np.ndarray,
    ) -> np.ndarray:
        """Computes ``odes`` wrt ``velocities`` inner-product wrt ``states``.

        Computes the gradient `∂<δ𝓡,∂𝓡/∂𝓥⋅δ𝓥>/∂𝓤`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_velocities: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.

        Returns:
            Gradient in shape `(NUM_VAR,num_radial,num_angular)``.
        """
        self._check_array(d_velocities, self._vertices.shape)
        self._check_array(d_odes, self._odes.shape)
        gradient = np.zeros_like(self._states)
        vec_jac_product = np.zeros_like(self._states)
        odes_wrt_states_pert = np.zeros_like(self._odes_wrt_states)
        step = 1e-6 * (1. + np.linalg.norm(self._velocities))**0.5 / np.linalg.norm(d_velocities)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_odes_wrt_states(
                self._mach, self._vertices, self._velocities + sign * step * part(d_velocities),
                self._states, odes_wrt_states_pert,
            )
            disc.apply_odes_wrt_states_rev(
                odes_wrt_states_pert, d_odes.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_odes_wrt_velocities_inner_product_wrt_vertices(
        self,
        d_odes: np.ndarray,
        d_velocities: np.ndarray,
    ) -> np.ndarray:
        """Computes ``odes`` wrt ``velocities`` inner-product wrt ``vertices``.

        Computes the gradient `∂<δ𝓡,∂𝓡/∂𝓥⋅δ𝓥>/∂𝓧`.

        Args:
            d_odes: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.
            d_velocities: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.

        Returns:
            Gradient in shape `(NUM_DIM,num_radial+1,num_angular+1)``.
        """
        self._check_array(d_velocities, self._vertices.shape)
        self._check_array(d_odes, self._odes.shape)
        gradient = np.zeros_like(self._vertices)
        vec_jac_product = np.zeros_like(self._vertices)
        odes_wrt_vertices_pert = np.zeros_like(self._odes_wrt_vertices)
        step = 1e-6 * (1. + np.linalg.norm(self._velocities))**0.5 / np.linalg.norm(d_velocities)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_odes_wrt_vertices(
                self._mach, self._vertices, self._velocities + sign * step * part(d_velocities),
                self._states, odes_wrt_vertices_pert,
            )
            disc.apply_odes_wrt_vertices_rev(
                odes_wrt_vertices_pert, d_odes.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_forces_wrt_states_inner_product_wrt_states(
        self,
        d_forces: np.ndarray,
        d_states: np.ndarray,
    ) -> np.ndarray:
        """Computes ``forces`` wrt ``states`` inner-product wrt ``states``.

        Computes the gradient `∂<δ𝓕,∂𝓕/∂𝓤⋅δ𝓤>/∂𝓤`.

        Args:
            d_forces: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_angular)``.
            d_states: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.

        Returns:
            Gradient in shape `(NUM_VAR,num_radial,num_angular)``.
        """
        self._check_array(d_states, self._states.shape)
        self._check_array(d_forces, self._forces.shape)
        gradient = np.zeros_like(self._states)
        vec_jac_product = np.zeros_like(self._states)
        forces_wrt_states_pert = np.zeros_like(self._forces_wrt_states)
        step = 1e-6 * (1. + np.linalg.norm(self._states))**0.5 / np.linalg.norm(d_states)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_forces_wrt_states(
                self._vertices, self._states + sign * step * part(d_states),
                forces_wrt_states_pert,
            )
            disc.apply_forces_wrt_states_rev(
                forces_wrt_states_pert, d_forces.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_forces_wrt_states_inner_product_wrt_vertices(
        self,
        d_forces: np.ndarray,
        d_states: np.ndarray,
    ) -> np.ndarray:
        """Computes ``forces`` wrt ``states`` inner-product wrt ``vertices``.

        Computes the gradient `∂<δ𝓕,∂𝓕/∂𝓤⋅δ𝓤>/∂𝓧`.

        Args:
            d_forces: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_angular)``.
            d_states: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_VAR,num_radial,num_angular)``.

        Returns:
            Gradient in shape `(NUM_DIM,num_radial+1,num_angular+1)``.
        """
        self._check_array(d_states, self._states.shape)
        self._check_array(d_forces, self._forces.shape)
        gradient = np.zeros_like(self._vertices)
        vec_jac_product = np.zeros_like(self._vertices)
        forces_wrt_vertices_pert = np.zeros_like(self._forces_wrt_vertices)
        step = 1e-6 * (1. + np.linalg.norm(self._states))**0.5 / np.linalg.norm(d_states)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_forces_wrt_vertices(
                self._vertices, self._states + sign * step * part(d_states),
                forces_wrt_vertices_pert,
            )
            disc.apply_forces_wrt_vertices_rev(
                forces_wrt_vertices_pert, d_forces.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_forces_wrt_vertices_inner_product_wrt_states(
        self,
        d_forces: np.ndarray,
        d_vertices: np.ndarray,
    ) -> np.ndarray:
        """Computes ``forces`` wrt ``vertices`` inner-product wrt ``states``.

        Computes the gradient `∂<δ𝓕,∂𝓕/∂𝓧⋅δ𝓧>/∂𝓤`.

        Args:
            d_forces: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_angular)``.
            d_vertices: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.

        Returns:
            Gradient in shape `(NUM_VAR,num_radial,num_angular)``.
        """
        self._check_array(d_vertices, self._vertices.shape)
        self._check_array(d_forces, self._forces.shape)
        gradient = np.zeros_like(self._states)
        vec_jac_product = np.zeros_like(self._states)
        forces_wrt_states_pert = np.zeros_like(self._forces_wrt_states)
        step = 1e-6 * (1. + np.linalg.norm(self._vertices))**0.5 / np.linalg.norm(d_vertices)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_forces_wrt_states(
                self._vertices + sign * step * part(d_vertices), self._states,
                forces_wrt_states_pert,
            )
            disc.apply_forces_wrt_states_rev(
                forces_wrt_states_pert, d_forces.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient

    def compute_forces_wrt_vertices_inner_product_wrt_vertices(
        self,
        d_forces: np.ndarray,
        d_vertices: np.ndarray,
    ) -> np.ndarray:
        """Computes ``forces`` wrt ``vertices`` inner-product wrt ``vertices``.

        Computes the gradient `∂<δ𝓕,∂𝓕/∂𝓧⋅δ𝓧>/∂𝓧`.

        Args:
            d_forces: Covector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_angular)``.
            d_vertices: Vector to multiply to the Jacobians. Must be complex FORTRAN-contiguous
                array in shape ``(NUM_DIM,num_radial+1,num_angular+1)``.

        Returns:
            Gradient in shape `(NUM_DIM,num_radial+1,num_angular+1)``.
        """
        self._check_array(d_vertices, self._vertices.shape)
        self._check_array(d_forces, self._forces.shape)
        gradient = np.zeros_like(self._vertices)
        vec_jac_product = np.zeros_like(self._vertices)
        forces_wrt_vertices_pert = np.zeros_like(self._forces_wrt_vertices)
        step = 1e-6 * (1. + np.linalg.norm(self._vertices))**0.5 / np.linalg.norm(d_vertices)
        for sign, (part, factor) in product((1., -1.), ((np.real, 1.), (np.imag, 1.j))):
            disc.compute_forces_wrt_vertices(
                self._vertices + sign * step * part(d_vertices), self._states,
                forces_wrt_vertices_pert,
            )
            disc.apply_forces_wrt_vertices_rev(
                forces_wrt_vertices_pert, d_forces.conj(), vec_jac_product)
            gradient += sign * factor * vec_jac_product
        gradient = gradient / 2. / step
        return gradient
