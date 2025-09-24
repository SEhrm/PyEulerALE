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

The `SpatialDiscretization` Python class stores the cell averaged states $`\underline{\pmb{u}}`$,
the grid vertices $`\underline{\vec{x}}`$, the grid velocities $`\underline{\vec{v}}`$ and the
airfoil section forces $`\underline{\vec{f}}`$ and provides the operators for

* the nonlinear autonomous ordinary differential equation

$$
\frac{\mathrm{d}\underline{\pmb{u}}}{\mathrm{d}t} =
\underline{\pmb{r}}(\underline{\pmb{u}}, \underline{\vec{x}}, \underline{\vec{v}}),
\quad
\underline{\vec{f}} =
\underline{\vec{f}}(\underline{\pmb{u}}, \underline{\vec{x}})\text{,}
$$

* the continuous-time time-invariant linearized state-space representation

$$
\frac{\mathrm{d}\delta\underline{\pmb{u}}}{\mathrm{d}t} =
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\pmb{u}}}\cdot\delta\underline{\pmb{u}} +
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\vec{x}}}\cdot\delta\underline{\vec{x}} +
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\vec{v}}}\cdot\delta\underline{\vec{v}},
\quad
\underline{\vec{f}} =
\frac{\partial\underline{\vec{f}}}{\partial\underline{\pmb{u}}}\cdot\delta\underline{\pmb{u}} +
\frac{\partial\underline{\vec{f}}}{\partial\underline{\vec{x}}}\cdot\delta\underline{\vec{x}}
\text{,}
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

## Usage

The spatial discretization can be initialized by

```python
from py_euler_ale import SpatialDiscretization

solver = SpatialDiscretization(
  grid_file='path/to/grid.plot3d',
  mach_numer=0.5,
  angle_of_attack=1.25,
  rusanov_factor=1e-1,
)
```

* Only two-dimensional, structured, closed, O-type grids in can be processed ; the grid file needs
  to be supplied in PLOT3D format with a single block.

  <details> <summary>Example</summary>

  For example, a grid with $`m`$ vertices defining the airfoil and $`n`$ layers would read

  ```text
  1
  𝑚 𝑛
  𝑥₁₁
  𝑥₁₂
  ⋮
  𝑥₁ₙ
  𝑥₂₁
  𝑥₂₂
  ⋮
  𝑥ₘₙ
  𝑧₁₁
  𝑧₁₂
  𝑧ₘₙ
  ```

  The first index going radially outward and the second index going angular around the airfoil; for
  closure, the points need to satisfy $`(x_{i1},y_{i1}) = (x_{in},y_{in}) \forall i=1,\ldots,m`$.

  </details>

* The free-stream Mach number should be chosen below the critical Mach number for the airfoil. The
  Scheme is unlikely to produce reliable results for shocks in the sonic regine.
* The far-field angle of attack will be read in degrees.
* The Rusanov/Lax-Friedrich should be chosen typically between 0 and 1. Higher values increase
  stability and prevent oscillations in the solution but introduce numerical dissipation decreasing
  the accuracy.

## Examples

### NACA-0004 Frequency Response

Consider an airfoil at zero far-field angle-of-attack undergoing small-amplitude oscillations in
pitch angle around some grid coordinates $`\vec{x}_\text{a} := (x_\text{a}, 0)`$.
To compute the frequency response, first the Euler equations are solved for the steady-state by
iterating the nonlinear ordinary differential equation with pseudo-transient continuation (PTC).
The global pseudo time-step size is controlled by switched evolution relaxation (SER) where the
time-step size is inverse proportional to the residual norm (note the use of the resolvent):

$$
\underline{\pmb{u}}^{n+1} = \underline{\pmb{u}}^{n} - \bigg(
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\pmb{u}}} - \sigma^n\text{Id}
\bigg)^{-1}
\underline{\pmb{r}}(\underline{\pmb{u}}^n, \underline{\vec{x}}, \underline{\vec{0}})
$$

At the steady-state, the transfer functions from pitch angle $`\alpha`$ to coefficients of lift and
moment at Laplace variable $`s`$ read

$$
\frac{\mathcal{L}(c_\text{l},c_\text{m})}{\mathcal{L}\alpha} :=
\Bigg[
\frac{\partial(c_\text{l},c_\text{m})}{\partial\underline{\vec{f}}}
\Bigg(
\frac{\partial\underline{\vec{f}}}{\partial\underline{\vec{x}}} -
\frac{\partial\underline{\vec{f}}}{\partial\underline{\pmb{u}}}
\bigg(
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\pmb{u}}} - s\text{Id}
\bigg)^{-1}
\bigg(
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\vec{x}}} +
\frac{\partial\underline{\pmb{r}}}{\partial\underline{\vec{v}}} s
\bigg)
\Bigg) +
\frac{\partial(c_\text{l},c_\text{m})}{\partial\underline{\vec{x}}}
\Bigg]
\frac{\partial\underline{\vec{x}}}{\partial\alpha}
\text{.}
$$

The gain $`\partial\underline{\vec{x}}\textfractionsolidus\partial\alpha`$ follows from the rotation
around
$`\vec{x}_\text{a}`$ and the gains $`\partial(c_\text{l},c_\text{m})\textfractionsolidus
\partial\underline{\vec{f}}`$ and $`\partial(c_\text{l},c_\text{m})\textfractionsolidus
\partial\underline{\vec{x}}`$ follow from the definition of the (classical) coefficients of section
lift and moment

$$
c_\text{l} := \dfrac{\oint f_z \mathrm{d}s}{\varrho_\infty u_\infty c\textfractionsolidus 2}, \quad
c_\text{m} := \dfrac{\oint f_x z - f_z (x - x_\text{a}) \mathrm{d}s}{\varrho_\infty u_\infty c^2\textfractionsolidus 2}\text.
$$

Running

```commandline
python examples/response/run.py examples/grids/naca0004_129x385.plot3d 1.008930 0.5 0.0
```

computes the frequency responses of the NACA-0004, with its chord measuring $`1.008930`$ grid units,
in free-stream Mach number $`Ma_\infty=0.5`$, pitching around the leading edge, at various values
of reduced frequency $`(\omega c)\textfractionsolidus(2 u_\infty)`$. The results compare well to the
results from Jordan ["_Aerodynamic flutter coefficients for subsonic, sonic and supersonic flow
(linear two-dimensional theory)_"](https://reports.aerade.cranfield.ac.uk/handle/1826.2/3495) (Note
the different definitions of frequency and lift/moment!)

<p align="center">
  <img src=examples/response/bode.png>
</p>

## Copyright

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
