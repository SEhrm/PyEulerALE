#!/usr/bin/env python

"""Unittests for the linearization features of ``SpatialDiscretization``

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
"""

import unittest
from pathlib import Path

import numpy as np

from py_euler_ale import SpatialDiscretization


# ruff: noqa: SLF001
class TestJacobi(unittest.TestCase):
    """Tests for the Jacobians"""

    def setUp(self) -> None:
        """Preparation done for each test

        Creates a fresh ``SpatialDiscretization`` instance, sets the ``states`` randomly, but
        similar to the free-stream, compute ``odes`` and ``forces``, and perform linearization.
        """
        self.rng = np.random.default_rng(seed=1)
        self.solver = SpatialDiscretization(
            grid_file=Path(__file__).parent / Path("naca0012_8x9.plot3d"),
        )
        self.solver.states[:] *= (self.rng.random(self.solver.states.shape) - 0.5) * 0.01 + 1.
        self.solver.compute_odes()
        self.solver.compute_forces()
        self.solver.linearize()

    def random_like(self, array: np.ndarray) -> np.ndarray:
        """Creates a randomly populated complex FORTRAN-contiguous array

        Args:
            array: Array whose shape to copy.

        Returns:
            Random array
        """
        d_states = np.empty_like(array, order="F", dtype=complex)
        d_states[:] = self.rng.random(array.shape) + self.rng.random(array.shape) * 1j
        return d_states

    def test_odes_wrt_mach(self) -> None:
        """Compare Jacobians of ``odes`` wrt ``mach`` with finite-difference"""
        odes_0 = self.solver.odes.copy()
        jacobi_fd = np.zeros((*odes_0.shape, 1))
        self.solver.mach_number += 1e-7
        self.solver.compute_odes()
        jacobi_fd[:, :, :, 0] = (self.solver.odes - odes_0) / 1e-7
        # testing jacobians
        for m in range(self.solver.num_radial):
            for n in range(self.solver.num_angular):
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :],
                    self.solver._odes_wrt_mach[:, :, m, n],
                    atol=1e-6, rtol=1e-5,
                )
        # testing fwd
        d_mach = np.asfortranarray(1. + 2.j)
        np.testing.assert_allclose(
            np.einsum("imnj,j->imn", jacobi_fd, d_mach),
            self.solver.apply_odes_wrt_mach_fwd(d_mach),
            atol=1e-6, rtol=1e-5,
        )
        # testing rev
        d_odes = self.random_like(odes_0)
        np.testing.assert_allclose(
            np.einsum("imnj,imn->j", jacobi_fd, d_odes),
            self.solver.apply_odes_wrt_mach_rev(d_odes),
            atol=1e-6, rtol=1e-5,
        )

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
        # testing jacobians
        for m in range(self.solver.num_radial):
            for n in range(self.solver.num_angular):
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, n],
                    self.solver._odes_wrt_states[:, :, m, n, 0],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, (n - 1) % self.solver.num_angular],
                    self.solver._odes_wrt_states[:, :, m, n, 1],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, (n - 2) % self.solver.num_angular],
                    self.solver._odes_wrt_states[:, :, m, n, 2],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, (n + 1) % self.solver.num_angular],
                    self.solver._odes_wrt_states[:, :, m, n, 3],
                    atol=1e-6, rtol=1e-5,
                )
                np.testing.assert_allclose(
                    jacobi_fd[:, m, n, :, m, (n + 2) % self.solver.num_angular],
                    self.solver._odes_wrt_states[:, :, m, n, 4],
                    atol=1e-6, rtol=1e-5,
                )
                if m > 0:
                    np.testing.assert_allclose(
                        jacobi_fd[:, m, n, :, m - 1, n],
                        self.solver._odes_wrt_states[:, :, m, n, 5],
                        atol=1e-6, rtol=1e-5,
                    )
                if m > 1:
                    np.testing.assert_allclose(
                        jacobi_fd[:, m, n, :, m - 2, n],
                        self.solver._odes_wrt_states[:, :, m, n, 6],
                        atol=1e-6, rtol=1e-5,
                    )
                if m < self.solver.num_radial - 1:
                    np.testing.assert_allclose(
                        jacobi_fd[:, m, n, :, m + 1, n],
                        self.solver._odes_wrt_states[:, :, m, n, 7],
                        atol=1e-6, rtol=1e-5,
                    )
                if m < self.solver.num_radial - 2:
                    np.testing.assert_allclose(
                        jacobi_fd[:, m, n, :, m + 2, n],
                        self.solver._odes_wrt_states[:, :, m, n, 8],
                        atol=1e-6, rtol=1e-5,
                    )
        # testing fwd
        d_states = self.random_like(states_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,jkl->imn", jacobi_fd, d_states),
            self.solver.apply_odes_wrt_states_fwd(d_states),
            atol=1e-6, rtol=1e-5,
        )
        # testing rev
        d_odes = self.random_like(odes_0)
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
        # testing jacobians
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
        d_vertices = self.random_like(vertices_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,jkl->imn", jacobi_fd, d_vertices),
            self.solver.apply_odes_wrt_vertices_fwd(d_vertices),
            atol=1e-6, rtol=1e-5,
        )
        # testing rev
        d_odes = self.random_like(odes_0)
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
        # testing jacobians
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
        d_velocities = self.random_like(velocities_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,jkl->imn", jacobi_fd, d_velocities),
            self.solver.apply_odes_wrt_velocities_fwd(d_velocities),
            atol=1e-6, rtol=1e-5,
        )
        # testing rev
        d_odes = self.random_like(odes_0)
        np.testing.assert_allclose(
            np.einsum("imnjkl,imn->jkl", jacobi_fd, d_odes),
            self.solver.apply_odes_wrt_velocities_rev(d_odes),
            atol=1e-6, rtol=1e-5,
        )

    def test_forces_wrt_states(self) -> None:
        """Compare Jacobians of ``forces`` wrt ``states`` with finite-difference"""
        states_0 = self.solver.states.copy()
        forces_0 = self.solver.forces.copy()
        jacobi_fd = np.zeros((*forces_0.shape, *states_0.shape))
        for i in range(4):
            for m in range(self.solver.num_radial):
                for n in range(self.solver.num_angular):
                    self.solver.states[:] = np.copy(states_0)
                    self.solver.states[i, m, n] += 1e-7
                    self.solver.compute_forces()
                    jacobi_fd[:, :, i, m, n] \
                        = (self.solver.forces - forces_0) / 1e-7
        # testing jacobians
        for n in range(self.solver.num_angular):
            np.testing.assert_allclose(
                jacobi_fd[:, n, :, 0, n],
                self.solver._forces_wrt_states[:, :, n],
                atol=1e-6, rtol=1e-5,
            )
        # testing fwd
        d_states = self.random_like(states_0)
        np.testing.assert_allclose(
            np.einsum("injkl,jkl->in", jacobi_fd, d_states),
            self.solver.apply_forces_wrt_states_fwd(d_states),
            atol=1e-6, rtol=1e-5,
        )
        # testing bwd
        d_forces = self.random_like(forces_0)
        np.testing.assert_allclose(
            np.einsum("injkl,in->jkl", jacobi_fd, d_forces),
            self.solver.apply_forces_wrt_states_rev(d_forces),
            atol=1e-6, rtol=1e-5,
        )

    def test_forces_wrt_vertices(self) -> None:
        """Compare Jacobians of ``forces`` wrt ``vertices`` with finite-difference"""
        vertices_0 = self.solver.vertices.copy()
        forces_0 = self.solver.forces.copy()
        jacobi_fd = np.zeros((*forces_0.shape, *self.solver._vertices.shape))
        for i in range(2):
            for m in range(self.solver.num_radial + 1):
                for n in range(self.solver.num_angular + 1):
                    self.solver.vertices[:] = np.copy(vertices_0)
                    self.solver.vertices[i, m, n] += 1e-8
                    self.solver.compute_forces()
                    jacobi_fd[:, :, i, m, n] \
                        = (self.solver.forces - forces_0) / 1e-8
        # testing jacobians
        for n in range(self.solver.num_angular):
            np.testing.assert_allclose(
                jacobi_fd[:, n, :, 0, n],
                self.solver._forces_wrt_vertices[:, :, n, 0],
                atol=1e-6, rtol=1e-5,
            )
            np.testing.assert_allclose(
                jacobi_fd[:, n, :, 0, n + 1],
                self.solver._forces_wrt_vertices[:, :, n, 1],
                atol=1e-6, rtol=1e-5,
            )
        # testing fwd
        d_vertices = self.random_like(vertices_0)
        np.testing.assert_allclose(
            np.einsum("injkl,jkl->in", jacobi_fd, d_vertices),
            self.solver.apply_forces_wrt_vertices_fwd(d_vertices),
            atol=1e-6, rtol=1e-5,
        )
        # testing rev
        d_forces = self.random_like(forces_0)
        np.testing.assert_allclose(
            np.einsum("injkl,in->jkl", jacobi_fd, d_forces),
            self.solver.apply_forces_wrt_vertices_rev(d_forces),
            atol=1e-6, rtol=1e-5,
        )

    def test_solve(self) -> None:
        """Check solutions for Jacobians of ``states`` wrt ``odes``"""
        shift = self.rng.random() + self.rng.random() * 1j
        # testing fwd
        d_states = self.random_like(self.solver.states)
        np.testing.assert_allclose(
            self.solver.solve_odes_wrt_states_fwd(
                d_odes=self.solver.apply_odes_wrt_states_fwd(
                    d_states=d_states,
                ) - shift * d_states,
                shift=shift,
            ),
            d_states,
        )
        # testing adj
        d_odes = self.random_like(self.solver.odes)
        np.testing.assert_allclose(
            self.solver.solve_odes_wrt_states_adj(
                d_states=self.solver.apply_odes_wrt_states_rev(
                    d_odes=d_odes,
                ) - shift.conjugate() * d_odes,
                shift=shift,
            ),
            d_odes,
        )

    def test_odes_wrt_states_inner_product_wrt_mach(self) -> None:
        """Checks gradient of ``odes`` wrt ``states`` inner-product wrt ``mach_number``"""
        d_states = self.random_like(self.solver.states)
        d_odes = self.random_like(self.solver.odes)
        gradient = np.zeros((1,), dtype=complex)
        mach_0 = self.solver.mach_number
        dot_0 = np.vdot(d_odes, self.solver.apply_odes_wrt_states_fwd(d_states))
        self.solver.mach_number = mach_0 + 1e-6
        self.solver.linearize()
        gradient[0] = (np.vdot(
            d_odes, self.solver.apply_odes_wrt_states_fwd(d_states),
        ) - dot_0) / 1e-6
        self.solver.mach_number = mach_0
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_odes_wrt_states_inner_product_wrt_mach(
                d_odes, d_states,
            ), gradient, atol=1e-6, rtol=1e-5,
        )

    def test_odes_wrt_states_inner_product_wrt_states(self) -> None:
        """Checks gradient of ``odes`` wrt ``states`` inner-product wrt ``states``"""
        d_states = self.random_like(self.solver.states)
        d_odes = self.random_like(self.solver.odes)
        gradient = np.zeros_like(self.solver.states, dtype=complex)
        states_0 = self.solver.states.copy()
        dot_0 = np.vdot(d_odes, self.solver.apply_odes_wrt_states_fwd(d_states))
        for i in range(4):
            for m in range(self.solver.num_radial):
                for n in range(self.solver.num_angular):
                    self.solver.states[:] = np.copy(states_0)
                    self.solver.states[i, m, n] += 1e-6
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_odes, self.solver.apply_odes_wrt_states_fwd(d_states),
                    ) - dot_0) / 1e-6
        self.solver.states[:] = np.copy(states_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_odes_wrt_states_inner_product_wrt_states(
                d_odes, d_states,
            ), gradient, atol=1e-6, rtol=1e-5,
        )

    def test_odes_wrt_states_inner_product_wrt_vertices(self) -> None:
        """Checks gradient of ``odes`` wrt ``states`` inner-product wrt ``vertices``"""
        d_states = self.random_like(self.solver.states)
        d_odes = self.random_like(self.solver.odes)
        gradient = np.zeros_like(self.solver.vertices, dtype=complex)
        vertices_0 = self.solver.vertices.copy()
        dot_0 = np.vdot(d_odes, self.solver.apply_odes_wrt_states_fwd(d_states))
        for i in range(2):
            for m in range(self.solver.num_radial + 1):
                for n in range(self.solver.num_angular + 1):
                    self.solver.vertices[:] = np.copy(vertices_0)
                    step = self.solver.vertices[i, m, n].real * 1e-6 + 1e-6
                    self.solver.vertices[i, m, n] += step
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_odes, self.solver.apply_odes_wrt_states_fwd(d_states),
                    ) - dot_0) / step
        self.solver.vertices[:] = np.copy(vertices_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_odes_wrt_states_inner_product_wrt_vertices(
                d_odes, d_states,
            ), gradient, atol=1e-5, rtol=1e-4,
        )

    def test_odes_wrt_vertices_inner_product_wrt_mach(self) -> None:
        """Checks gradient of ``odes`` wrt ``vertices`` inner-product wrt ``mach_number``"""
        d_vertices = self.random_like(self.solver.vertices)
        d_odes = self.random_like(self.solver.odes)
        gradient = np.zeros((1,), dtype=complex)
        mach_0 = self.solver.mach_number
        dot_0 = np.vdot(d_odes, self.solver.apply_odes_wrt_vertices_fwd(d_vertices))
        self.solver.mach_number = mach_0 + 1e-6
        self.solver.linearize()
        gradient[0] = (np.vdot(
            d_odes, self.solver.apply_odes_wrt_vertices_fwd(d_vertices),
        ) - dot_0) / 1e-6
        self.solver.mach_number = mach_0
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_odes_wrt_vertices_inner_product_wrt_mach(
                d_odes, d_vertices,
            ), gradient, atol=1e-6, rtol=1e-5,
        )

    def test_odes_wrt_vertices_inner_product_wrt_states(self) -> None:
        """Checks gradient of ``odes`` wrt ``vertices`` inner-product wrt ``states``"""
        d_vertices = self.random_like(self.solver.vertices)
        d_odes = self.random_like(self.solver.odes)
        gradient = np.zeros_like(self.solver.states, dtype=complex)
        states_0 = self.solver.states.copy()
        dot_0 = np.vdot(d_odes, self.solver.apply_odes_wrt_vertices_fwd(d_vertices))
        for i in range(4):
            for m in range(self.solver.num_radial):
                for n in range(self.solver.num_angular):
                    self.solver.states[:] = np.copy(states_0)
                    self.solver.states[i, m, n] += 1e-6
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_odes, self.solver.apply_odes_wrt_vertices_fwd(d_vertices),
                    ) - dot_0) / 1e-6
        self.solver.states[:] = np.copy(states_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_odes_wrt_vertices_inner_product_wrt_states(
                d_odes, d_vertices,
            ), gradient, atol=1e-6, rtol=5e-5,
        )

    def test_odes_wrt_vertices_inner_product_wrt_vertices(self) -> None:
        """Checks gradient of ``odes`` wrt ``vertices`` inner-product wrt ``vertices``"""
        d_vertices = self.random_like(self.solver.vertices)
        d_odes = self.random_like(self.solver.odes)
        gradient = np.zeros_like(self.solver.vertices, dtype=complex)
        vertices_0 = self.solver.vertices.copy()
        dot_0 = np.vdot(d_odes, self.solver.apply_odes_wrt_vertices_fwd(d_vertices))
        for i in range(2):
            for m in range(self.solver.num_radial + 1):
                for n in range(self.solver.num_angular + 1):
                    self.solver.vertices[:] = np.copy(vertices_0)
                    step = self.solver.vertices[i, m, n].real * 1e-6 + 1e-6
                    self.solver.vertices[i, m, n] += step
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_odes, self.solver.apply_odes_wrt_vertices_fwd(d_vertices),
                    ) - dot_0) / step
        self.solver.vertices[:] = np.copy(vertices_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_odes_wrt_vertices_inner_product_wrt_vertices(
                d_odes, d_vertices,
            ), gradient, atol=1e-5, rtol=1e-4,
        )

    def test_odes_wrt_velocities_inner_product_wrt_mach(self) -> None:
        """Checks gradient of ``odes`` wrt ``velocities`` inner-product wrt ``mach_number``"""
        d_velocities = self.random_like(self.solver.velocities)
        d_odes = self.random_like(self.solver.odes)
        gradient = np.zeros((1,), dtype=complex)
        mach_0 = self.solver.mach_number
        dot_0 = np.vdot(d_odes, self.solver.apply_odes_wrt_velocities_fwd(d_velocities))
        self.solver.mach_number = mach_0 + 1e-6
        self.solver.linearize()
        gradient[0] = (np.vdot(
            d_odes, self.solver.apply_odes_wrt_velocities_fwd(d_velocities),
        ) - dot_0) / 1e-6
        self.solver.mach_number = mach_0
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_odes_wrt_velocities_inner_product_wrt_mach(
                d_odes, d_velocities,
            ), gradient, atol=1e-6, rtol=1e-5,
        )

    def test_odes_wrt_velocities_inner_product_wrt_states(self) -> None:
        """Checks gradient of ``odes`` wrt ``velocities`` inner-product wrt ``states``"""
        d_velocities = self.random_like(self.solver.velocities)
        d_odes = self.random_like(self.solver.odes)
        gradient = np.zeros_like(self.solver.states, dtype=complex)
        states_0 = self.solver.states.copy()
        dot_0 = np.vdot(d_odes, self.solver.apply_odes_wrt_velocities_fwd(d_velocities))
        for i in range(4):
            for m in range(self.solver.num_radial):
                for n in range(self.solver.num_angular):
                    self.solver.states[:] = np.copy(states_0)
                    self.solver.states[i, m, n] += 1e-6
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_odes, self.solver.apply_odes_wrt_velocities_fwd(d_velocities),
                    ) - dot_0) / 1e-6
        self.solver.states[:] = np.copy(states_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_odes_wrt_velocities_inner_product_wrt_states(
                d_odes, d_velocities,
            ), gradient, atol=1e-6, rtol=5e-5,
        )

    def test_odes_wrt_velocities_inner_product_wrt_vertices(self) -> None:
        """Checks gradient of ``odes`` wrt ``velocities`` inner-product wrt ``vertices``"""
        d_velocities = self.random_like(self.solver.velocities)
        d_odes = self.random_like(self.solver.odes)
        gradient = np.zeros_like(self.solver.vertices, dtype=complex)
        vertices_0 = self.solver.vertices.copy()
        dot_0 = np.vdot(d_odes, self.solver.apply_odes_wrt_velocities_fwd(d_velocities))
        for i in range(2):
            for m in range(self.solver.num_radial + 1):
                for n in range(self.solver.num_angular + 1):
                    self.solver.vertices[:] = np.copy(vertices_0)
                    step = self.solver.vertices[i, m, n].real * 1e-6 + 1e-6
                    self.solver.vertices[i, m, n] += step
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_odes, self.solver.apply_odes_wrt_velocities_fwd(d_velocities),
                    ) - dot_0) / step
        self.solver.vertices[:] = np.copy(vertices_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_odes_wrt_velocities_inner_product_wrt_vertices(
                d_odes, d_velocities,
            ), gradient, atol=1e-5, rtol=1e-4,
        )

    def test_forces_wrt_states_inner_product_wrt_states(self) -> None:
        """Checks gradient of ``forces`` wrt ``states`` inner-product wrt ``states``"""
        d_states = self.random_like(self.solver.states)
        d_forces = self.random_like(self.solver.forces)
        gradient = np.zeros_like(self.solver.states, dtype=complex)
        states_0 = self.solver.states.copy()
        dot_0 = np.vdot(d_forces, self.solver.apply_forces_wrt_states_fwd(d_states))
        for i in range(4):
            for m in range(self.solver.num_radial):
                for n in range(self.solver.num_angular):
                    self.solver.states[:] = np.copy(states_0)
                    self.solver.states[i, m, n] += 1e-6
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_forces, self.solver.apply_forces_wrt_states_fwd(d_states),
                    ) - dot_0) / 1e-6
        self.solver.states[:] = np.copy(states_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_forces_wrt_states_inner_product_wrt_states(
                d_forces, d_states,
            ), gradient, atol=1e-6, rtol=1e-5,
        )

    def test_forces_wrt_states_inner_product_wrt_vertices(self) -> None:
        """Checks gradient of ``forces`` wrt ``states`` inner-product wrt ``vertices``"""
        d_states = self.random_like(self.solver.states)
        d_forces = self.random_like(self.solver.forces)
        gradient = np.zeros_like(self.solver.vertices, dtype=complex)
        vertices_0 = self.solver.vertices.copy()
        dot_0 = np.vdot(d_forces, self.solver.apply_forces_wrt_states_fwd(d_states))
        for i in range(2):
            for m in range(self.solver.num_radial + 1):
                for n in range(self.solver.num_angular + 1):
                    self.solver.vertices[:] = np.copy(vertices_0)
                    step = self.solver.vertices[i, m, n].real * 1e-6 + 1e-6
                    self.solver.vertices[i, m, n] += step
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_forces, self.solver.apply_forces_wrt_states_fwd(d_states),
                    ) - dot_0) / step
        self.solver.vertices[:] = np.copy(vertices_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_forces_wrt_states_inner_product_wrt_vertices(
                d_forces, d_states,
            ), gradient, atol=1e-5, rtol=1e-4,
        )

    def test_forces_wrt_vertices_inner_product_wrt_states(self) -> None:
        """Checks gradient of ``forces`` wrt ``vertices`` inner-product wrt ``states``"""
        d_vertices = self.random_like(self.solver.vertices)
        d_forces = self.random_like(self.solver.forces)
        gradient = np.zeros_like(self.solver.states, dtype=complex)
        states_0 = self.solver.states.copy()
        dot_0 = np.vdot(d_forces, self.solver.apply_forces_wrt_vertices_fwd(d_vertices))
        for i in range(4):
            for m in range(self.solver.num_radial):
                for n in range(self.solver.num_angular):
                    self.solver.states[:] = np.copy(states_0)
                    self.solver.states[i, m, n] += 1e-6
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_forces, self.solver.apply_forces_wrt_vertices_fwd(d_vertices),
                    ) - dot_0) / 1e-6
        self.solver.states[:] = np.copy(states_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_forces_wrt_vertices_inner_product_wrt_states(
                d_forces, d_vertices,
            ), gradient, atol=1e-6, rtol=1e-5,
        )

    def test_forces_wrt_vertices_inner_product_wrt_vertices(self) -> None:
        """Checks gradient of ``forces`` wrt ``vertices`` inner-product wrt ``vertices``"""
        d_vertices = self.random_like(self.solver.vertices)
        d_forces = self.random_like(self.solver.forces)
        gradient = np.zeros_like(self.solver.vertices, dtype=complex)
        vertices_0 = self.solver.vertices.copy()
        dot_0 = np.vdot(d_forces, self.solver.apply_forces_wrt_vertices_fwd(d_vertices))
        for i in range(2):
            for m in range(self.solver.num_radial + 1):
                for n in range(self.solver.num_angular + 1):
                    self.solver.vertices[:] = np.copy(vertices_0)
                    step = self.solver.vertices[i, m, n].real * 1e-6 + 1e-6
                    self.solver.vertices[i, m, n] += step
                    self.solver.linearize()
                    gradient[i, m, n] = (np.vdot(
                        d_forces, self.solver.apply_forces_wrt_vertices_fwd(d_vertices),
                    ) - dot_0) / step
        self.solver.vertices[:] = np.copy(vertices_0)
        self.solver.linearize()
        np.testing.assert_allclose(
            self.solver.compute_forces_wrt_vertices_inner_product_wrt_vertices(
                d_forces, d_vertices,
            ), gradient, atol=1e-5, rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
