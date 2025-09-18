#!/usr/bin/env python

"""Unittests for the linearization features of ``SpatialDiscretization``

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

import unittest
from pathlib import Path

import numpy as np

from py_euler_ale import SpatialDiscretization

rng = np.random.default_rng(seed=0)


def random_like(array: np.ndarray) -> np.ndarray:
    """Creates a randomly populated complex FORTRAN-contiguous array

    Args:
        array: Array whose shape to copy.

    Returns:
        Random array
    """
    d_states = np.empty_like(array, order="F", dtype=complex)
    d_states[:] = rng.random(array.shape) + rng.random(array.shape) * 1j
    return d_states


# ruff: noqa: SLF001
class TestJacobi(unittest.TestCase):
    """Tests for the Jacobians"""

    def setUp(self) -> None:
        """Preparation done for each test

        Creates a fresh ``SpatialDiscretization`` instance, sets the ``states`` randomly, but
        similar to the free-stream, compute ``odes`` and ``total_force``, and perform linearization.
        """
        self.solver = SpatialDiscretization(
            grid_file=Path(__file__).parent / Path("8x9.x"),
            rusanov_factor=1.,
        )
        self.solver.states[:] *= (rng.random(self.solver.states.shape) - 0.5) * 0.01 + 1.
        self.solver.compute_odes()
        self.solver.compute_total_force()
        self.solver.linearize()

    def test_odes_wrt_states(self) -> None:
        """Compare Jacobians of ``odes`` wrt ``states`` with finite-difference"""
        states_0 = self.solver.states.copy()
        odes_0 = self.solver.odes.copy()
        jacobi_fd = np.zeros((*odes_0.shape, *odes_0.shape))
        for i in range(4):
            for m in range(self.solver.num_radial):
                for n in range(self.solver.num_angular):
                    self.solver.states[:] = np.copy(states_0)
                    self.solver.states[i, m, n] += 1e-7
                    self.solver.compute_odes()
                    jacobi_fd[:, :, :, i, m, n] = (self.solver.odes - odes_0) / 1e-7
        # testing jacobis
        for m in range(self.solver.num_radial):
            for n in range(self.solver.num_angular):
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, n],
                    self.solver._odes_wrt_states[:, :, m, n, 2],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, (n - 1) % self.solver.num_angular],
                    self.solver._odes_wrt_states[:, :, m, n, 0],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, (n + 1) % self.solver.num_angular],
                    self.solver._odes_wrt_states[:, :, m, n, 4],
                    atol=1e-6, rtol=1e-5,
                )
                if m > 0:
                    np.testing.assert_allclose(
                        jacobi_fd[:, m, n, :, m - 1, n],
                        self.solver._odes_wrt_states[:, :, m, n, 1],
                        atol=1e-6, rtol=1e-5,
                    )
                if m < self.solver.num_radial - 1:
                    np.testing.assert_allclose(
                        jacobi_fd[:, m, n, :, m + 1, n],
                        self.solver._odes_wrt_states[:, :, m, n, 3],
                        atol=1e-6, rtol=1e-5,
                    )
        # testing fwd
        d_states = random_like(states_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,jkl->imn", jacobi_fd, d_states),
            self.solver.apply_odes_wrt_states_fwd(d_states),
            atol=1e-6, rtol=1e-5,
        )
        # testing rev
        d_odes = random_like(odes_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,imn->jkl", jacobi_fd, d_odes),
            self.solver.apply_odes_wrt_states_rev(d_odes),
            atol=1e-6, rtol=1e-5,
        )

    def test_odes_wrt_vertices(self) -> None:
        """Compare Jacobians of ``odes`` wrt ``vertices`` with finite-difference"""
        vertices_0 = self.solver.vertices.copy()
        odes_0 = self.solver.odes.copy()
        jacobi_fd = np.zeros((*odes_0.shape, *self.solver._vertices.shape))
        for i in range(2):
            for m in range(self.solver.num_radial + 1):
                for n in range(self.solver.num_angular + 1):
                    self.solver.vertices[:] = np.copy(vertices_0)
                    self.solver.vertices[i, m, n] += 1e-8
                    self.solver.compute_odes()
                    jacobi_fd[:, :, :, i, m, n] = (self.solver.odes - odes_0) / 1e-8
        # testing jacobis
        for m in range(self.solver.num_radial):
            for n in range(self.solver.num_angular):
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, n],
                    self.solver._odes_wrt_vertices[:, :, m, n, 0],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, n + 1],
                    self.solver._odes_wrt_vertices[:, :, m, n, 1],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m + 1, n + 1],
                    self.solver._odes_wrt_vertices[:, :, m, n, 2],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m + 1, n],
                    self.solver._odes_wrt_vertices[:, :, m, n, 3],
                    atol=1e-6, rtol=1e-5,
                )
        # testing fwd
        d_vertices = random_like(vertices_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,jkl->imn", jacobi_fd, d_vertices),
            self.solver.apply_odes_wrt_vertices_fwd(d_vertices),
            atol=1e-6, rtol=1e-5,
        )
        # testing rev
        d_odes = random_like(odes_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,imn->jkl", jacobi_fd, d_odes),
            self.solver.apply_odes_wrt_vertices_rev(d_odes),
            atol=1e-6, rtol=1e-5,
        )

    def test_odes_wrt_velocities(self) -> None:
        """Compare Jacobians of ``odes`` wrt ``velocities`` with finite-difference"""
        velocities_0 = self.solver.velocities.copy()
        odes_0 = self.solver.odes.copy()
        jacobi_fd = np.zeros((*odes_0.shape, *velocities_0.shape))
        for i in range(2):
            for m in range(self.solver.num_radial + 1):
                for n in range(self.solver.num_angular + 1):
                    self.solver.velocities[:] = np.copy(velocities_0)
                    self.solver.velocities[i, m, n] += 1e-8
                    self.solver.compute_odes()
                    jacobi_fd[:, :, :, i, m, n] = (self.solver.odes - odes_0) / 1e-8
        # testing jacobis
        for m in range(self.solver.num_radial):
            for n in range(self.solver.num_angular):
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, n],
                    self.solver._odes_wrt_velocities[:, :, m, n, 0],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, n + 1],
                    self.solver._odes_wrt_velocities[:, :, m, n, 1],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m + 1, n + 1],
                    self.solver._odes_wrt_velocities[:, :, m, n, 2],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m + 1, n],
                    self.solver._odes_wrt_velocities[:, :, m, n, 3],
                    atol=1e-6, rtol=1e-5,
                )
        # testing fwd
        d_velocities = random_like(velocities_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,jkl->imn", jacobi_fd, d_velocities),
            self.solver.apply_odes_wrt_velocities_fwd(d_velocities),
            atol=1e-6, rtol=1e-5,
        )
        # testing rev
        d_odes = random_like(odes_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,imn->jkl", jacobi_fd, d_odes),
            self.solver.apply_odes_wrt_velocities_rev(d_odes),
            atol=1e-6, rtol=1e-5,
        )

    def test_total_force_wrt_states(self) -> None:
        """Compare Jacobians of ``total_force`` wrt ``states`` with finite-difference"""
        states_0 = self.solver.states.copy()
        total_force_0 = self.solver.total_force.copy()
        jacobi_fd = np.zeros((2, *states_0.shape))
        for i in range(4):
            for m in range(self.solver.num_radial):
                for n in range(self.solver.num_angular):
                    self.solver.states[:] = np.copy(states_0)
                    self.solver.states[i, m, n] += 1e-7
                    self.solver.compute_total_force()
                    jacobi_fd[:, i, m, n] \
                        = (self.solver.total_force - total_force_0) / 1e-7
        # testing jacobis
        np.testing.assert_allclose(
            jacobi_fd[:, :, 0, :],
            self.solver._total_force_wrt_states,
            atol=1e-6, rtol=1e-5,
        )
        # testing fwd
        d_states = random_like(states_0)
        np.testing.assert_allclose(
            np.einsum("kimn,imn->k", jacobi_fd, d_states),
            self.solver.apply_total_force_wrt_states_fwd(d_states),
            atol=1e-6, rtol=1e-5,
        )
        # testing bwd
        d_total_force = random_like(total_force_0)
        np.testing.assert_allclose(
            np.einsum("kimn,k->imn", jacobi_fd, d_total_force),
            self.solver.apply_total_force_wrt_states_rev(d_total_force),
            atol=1e-6, rtol=1e-5,
        )

    def test_total_force_wrt_vertices(self) -> None:
        """Compare Jacobians of ``total_force`` wrt ``vertices`` with finite-difference"""
        vertices_0 = self.solver.vertices.copy()
        total_force_0 = self.solver.total_force.copy()
        jacobi_fd = np.zeros((*total_force_0.shape, *self.solver._vertices.shape))
        for i in range(2):
            for m in range(self.solver.num_radial + 1):
                for n in range(self.solver.num_angular + 1):
                    self.solver.vertices[:] = np.copy(vertices_0)
                    self.solver.vertices[i, m, n] += 1e-8
                    self.solver.compute_total_force()
                    jacobi_fd[:, i, m, n] \
                        = (self.solver.total_force - total_force_0) / 1e-8
        # testing jacobis todo
        # testing fwd
        d_vertices = random_like(vertices_0)
        np.testing.assert_allclose(
            np.einsum("kimn,imn->k", jacobi_fd, d_vertices),
            self.solver.apply_total_force_wrt_vertices_fwd(d_vertices),
            atol=1e-6, rtol=1e-5,
        )
        # testing rev
        d_total_force = random_like(total_force_0)
        np.testing.assert_allclose(
            np.einsum("kimn,k->imn", jacobi_fd, d_total_force),
            self.solver.apply_total_force_wrt_vertices_rev(d_total_force),
            atol=1e-6, rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
