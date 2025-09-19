# PyEulerALE

Simple finite volume discretization for the 2D compressible Euler equations around moving airfoils
implemented in FORTRAN 2008 with Python interface.

This package implements a finite volume method for the spatial discretization of the two-dimensional
compressible Euler equations around moving airfoils in arbitrary Lagrangian-Eulerian formulation
(ALE). The discretization uses a central-scheme with Rusanov/Lax-Friedrich flux for structured
O-type grids.

## Features

* Moving grid capabilities with arbitrary deformation (ALE formulation).
  You can assign each grid
  vertex a custom velocity (subject to the geometric conservation law).
* Data management and program flow in Python, wrapping spatial discretization procedures implemented
  in FORTRAN 2008.
  You can implement your application in Python by interacting with the spatial discretization
  wrapped
  in a Python class.
* Computation of all Jacobians by complex-step for linearized state-space formulation.
  You can implement implicit time-integration methods and transfer function, and sensitivity
  analysis.

The `SpatialDiscretization` Python class stores the cell averaged states $\underline{\pmb{u}}$, the
grid vertices $\underline{\vec{x}}$, the grid velocities $\underline{\vec{v}}$ and the total section
force $\vec{f}$ and provides the operators for

* the nonlinear autonomous ordinary differential equation

$$
\frac{\mathrm{d}\underline{\pmb{u}}}{\mathrm{d}t} =
\underline{\pmb{r}}(\underline{\pmb{u}}, \underline{\vec{x}}, \underline{\vec{v}}),
\quad
\vec{f} =
\vec{f}(\underline{\pmb{u}}, \underline{\vec{x}})\text{,}
$$

* the continuous-time time-invariant linearized state-space representation

$$
\frac{\mathrm{d}\delta\underline{\pmb{u}}}{\mathrm{d}t} =
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\pmb{u}}}\cdot\delta\underline{\pmb{u}} +
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\vec{x}}}\cdot\delta\underline{\vec{x}} +
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\vec{v}}}\cdot\delta\underline{\vec{v}},
\quad
\delta\vec{f} =
\frac{\partial\vec{f}}{\partial\underline{\pmb{u}}}\cdot\delta\underline{\pmb{u}} +
\frac{\partial\vec{f}}{\partial\underline{\vec{x}}}\cdot\delta\underline{\vec{x}}\text{,}
$$

* and the resolvent

$$
\delta\underline{\pmb{u}} = \left(
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\pmb{u}}} -
\sigma \mathrm{Id}
\right)^{-1} \cdot \delta\underline{\pmb{r}}\text{.}
$$

## Installation

Simply clone the repository and install the package through `pip`. The FORTRAN files will
automatically be compiled and processed through `f2py`.

```commandline
pip install ./PyEulerAle
```

## Examples

### Steady-State NACA-0012

```commandline
python run.py ../meshes/129x129.x 0.5 1.25 0.0 1.008930
```

computes the steady-state flow around the NACA-0012 airfoil at constant far-field angle-of-attack
$\alpha_\infty$ and (downward) heave speed $\dot{h}$.
By linear airfoil theory, the (upward) section coefficient of lift should converge to

$$c_\text{l} = \dfrac{2\pi}{\sqrt{1 - Ma_\infty^2}} \cdot \left(
\frac{\alpha_\infty}{1\,\text{rad}} + \frac{\dot{h}}{\Vert \vec{v}_\infty \Vert} \right)$$

The following pressure distribution gets written to `cp.csv`

<p align="center">
  <img src=examples/steady/expected/cp.png>
</p>

## Copyright

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
