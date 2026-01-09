# PyEulerALE

Simple finite volume discretization for the 2D compressible Euler equations around moving airfoils
implemented in FORTRAN 2008 with Python interface.

<p align="center">
  <img src=examples/response/sinusoidal.gif>
</p>

This package implements a finite volume method for the spatial discretization of the two-dimensional
compressible Euler equations around moving airfoils in arbitrary Lagrangian-Eulerian formulation
(ALE). The discretization uses a central-scheme with artificial dissipation for structured
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
  You can implement implicit time-integration methods, transfer functions, and sensitivity
  analysis.

The `SpatialDiscretization` Python class stores the cell averaged states $`𝓤`$,
the grid vertices $`𝓧`$, the grid velocities $`𝓥`$ and the
airfoil section forces $`𝓕`$, and provides the operators for

* the nonlinear autonomous ordinary differential equation

$$
\frac{\mathrm{d}𝓤}{\mathrm{d}t} =
𝓡(Ma_\infty, 𝓤, 𝓧, 𝓥),
\quad
𝓕 =
𝓕(𝓤, 𝓧)\text{,}
$$

* the continuous-time time-invariant linearized state-space representation

$$
\frac{\mathrm{d}\delta𝓤}{\mathrm{d}t} =
\frac{\partial𝓡}{\partial𝓤}\cdot\delta𝓤 +
\frac{\partial𝓡}{\partial𝓧}\cdot\delta𝓧 +
\frac{\partial𝓡}{\partial𝓥}\cdot\delta𝓥,
\quad
𝓕 =
\frac{\partial𝓕}{\partial𝓤}\cdot\delta𝓤 +
\frac{\partial𝓕}{\partial𝓧}\cdot\delta𝓧
\text{,}
$$

* the resolvent

$$
\delta𝓤 = \left(
\frac{\partial𝓡}{\partial𝓤} -
\sigma \mathrm{Id}
\right)^{-1} \cdot \delta𝓡
\text{,}
$$

* and the second order gradients

$$
\begin{gathered}
\left.\frac{\partial
\left\langle \delta𝓡, \frac{\partial𝓡}{\partial𝓤}\,\delta𝓤 \right\rangle
}{\partial(Ma_\infty, 𝓤, 𝓧)}\right\vert_{\delta𝓡,\delta𝓤}
, \quad
\left.\frac{\partial
\left\langle \delta𝓡, \frac{\partial𝓡}{\partial𝓧}\,\delta𝓧 \right\rangle
}{\partial(Ma_\infty, 𝓤, 𝓧)}\right\vert_{\delta𝓡,\delta𝓧}
, \quad
\left.\frac{\partial
\left\langle \delta𝓡, \frac{\partial𝓡}{\partial𝓥}\,\delta𝓥 \right\rangle
}{\partial(Ma_\infty, 𝓤, 𝓧)}\right\vert_{\delta𝓡,\delta𝓥}
\\
\left.\frac{\partial
\left\langle \delta𝓕, \frac{\partial𝓕}{\partial𝓤}\,\delta𝓤 \right\rangle
}{\partial(𝓤, 𝓧)}\right\vert_{\delta𝓕,\delta𝓤}
, \quad
\left.\frac{\partial
\left\langle \delta𝓕, \frac{\partial𝓕}{\partial𝓧}\,\delta𝓧 \right\rangle
}{\partial(𝓤, 𝓧)}\right\vert_{\delta𝓕,\delta𝓧}
\text{.}
\end{gathered}
$$

## Installation

Simply clone the repository and install the package through `pip`. The FORTRAN files will
automatically be compiled and processed through `f2py`.

```commandline
pip install ./PyEulerALE
```

## Usage

The spatial discretization can be initialized by

```python
from py_euler_ale import SpatialDiscretization

solver = SpatialDiscretization(
  grid_file='path/to/grid.plot3d',
  mach_number=0.5,
  angle_of_attack=1.25,
)
```

* Only two-dimensional, structured, closed, O-type grids in can be processed; the grid file needs
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

  The first index going radially outward and the second index going angular around the airfoil in
  direction of the pitch axis, e.g. trailing edge - lower side - leading edge - upper side. For
  closure, the points need to satisfy $`(x_{i1},z_{i1}) = (x_{in},z_{in}) \forall i=1,\ldots,m`$.

  </details>

* The free-stream Mach number should be chosen below the critical Mach number for the airfoil. The
  scheme is unlikely to produce reliable results for shocks in the sonic regime.
* The far-field angle of attack will be read in degrees.

## Examples

### NACA-0004 Frequency Response

Consider an airfoil at zero far-field angle-of-attack undergoing small-amplitude oscillations in
pitch angle around some grid coordinates $`\vec{x}_\text{a} := (x_\text{a}, 0)`$.
To compute the frequency response, first the Euler equations are solved for the steady-state by
iterating the nonlinear ordinary differential equation with pseudo-transient continuation (PTC).
The global pseudo time-step size is controlled by switched evolution relaxation (SER) where the
time-step size is inverse proportional to the residual norm (note the use of the resolvent):

$$
𝓡_n :=
𝓡(𝓤_n, 𝓧, 𝟎), \quad
𝓤_{n+1} := 𝓤_{n} - \left(
\frac{\partial𝓡}{\partial𝓤} -
\Vert𝓡_n\Vert^{-1}\text{Id}
\right)^{-1} 𝓡_n
$$

At the steady-state, the transfer functions from pitch angle $`\alpha`$ to coefficients of lift and
moment at Laplace variable $`s`$ read

$$
\left.\frac{ℒ(c_\text{l},c_\text{m})}{ℒ\alpha}\right\vert_s =
\frac{\partial(c_\text{l},c_\text{m})}{\partial𝓧}
\frac{\mathrm{d}𝓧}{\mathrm{d}\alpha} +
\frac{\partial(c_\text{l},c_\text{m})}{\partial𝓕}
\left.\frac{ℒ𝓕}{ℒ\alpha}\right\vert_s
\text{,}
$$

where

$$
\left.\frac{ℒ𝓕}{ℒ\alpha}\right\vert_s =
\frac{\partial𝓕}{\partial𝓧}
\frac{\mathrm{d}𝓧}{\mathrm{d}\alpha} +
\frac{\partial𝓕}{\partial𝓤}
\left.\frac{ℒ𝓤}{ℒ\alpha}\right\vert_s,
\qquad
\left.\frac{ℒ𝓤}{ℒ\alpha}\right\vert_s =
-\left(
\frac{\partial𝓡}{\partial𝓤} - s\text{Id}
\right)^{-1}
\left(
\frac{\partial𝓡}{\partial𝓧} +
\frac{\partial𝓡}{\partial𝓥} s
\right)
\frac{\mathrm{d}𝓧}{\mathrm{d}\alpha}
$$

are the transfer functions from pitch angle to forces and state respectively.
The gain $`\mathrm{d}𝓧\textfractionsolidus\mathrm{d}\alpha`$ follows from the
rotation
around $`\vec{x}_\text{a}`$ and the gains
$`\partial(c_\text{l},c_\text{m})\textfractionsolidus\partial𝓕`$ and
$`\partial(c_\text{l},c_\text{m})\textfractionsolidus\partial𝓧`$ follow from the
definition of the (classical) coefficients of section lift and moment

$$
c_\text{l} := \dfrac{\oint f_z \mathrm{d}s}
{\dfrac{\varrho_\infty}{2} u_\infty^2 c}, \quad
c_\text{m} := \dfrac{\oint f_x z - f_z (x - x_\text{a}) \mathrm{d}s}
{\dfrac{\varrho_\infty}{2} u_\infty^2 c^2}\text{.}
$$

Running

```commandline
python examples/response/run.py examples/grids/naca0004_129x129.plot3d 10. 0.5 0.0
```

computes the frequency responses of the NACA-0004, with its chord measuring $`10.0`$ grid units,
in free-stream Mach number $`Ma_\infty=0.5`$, pitching around the leading edge, at various values
of reduced frequency $`(\omega c)\textfractionsolidus(2 u_\infty)`$. The results compare well to the
results from Jordan ["_Aerodynamic flutter coefficients for subsonic, sonic and supersonic flow
(linear two-dimensional theory)_"](https://reports.aerade.cranfield.ac.uk/handle/1826.2/3495) (Note
the different definitions of frequency and lift/moment!)

<p align="center">
  <img src=examples/response/bode.png>
</p>

The script also produces ``cp.gz`` containing the steady-state pressure coefficients

$$
c_\text{p} := \dfrac{p - p_\infty}{\varrho_\infty u_\infty^2\textfractionsolidus 2}
$$

and a frequency response $`ℒc_\text{p}\textfractionsolidusℒ\alpha`$ at
$`\omega = 2 u_\infty \textfractionsolidus c`$.
From that, the time-dependent sinusoidal steady-state for $`2^\circ`$-pitching can be computed as

$$
c_\text{p}^\text{sss}(t) = c_\text{p} + 2^\circ \cdot
\left| \frac{ℒc_\text{p}}{ℒ\alpha} \right| \cdot
\cos\left( \omega t + \angle\left( \frac{ℒc_\text{p}}{ℒ\alpha} \right)\right)
$$

without the need for time-accurate simulation.
Note the difference in phase: lift is no longer governed by the pitch angle, but by the pitch rate.

### NACA-0012 Static Gain Derivative

The derivative with respect to the Mach number of the static gain of the lift coefficient with
respect to pitching reads

$$
\frac{\mathrm{d}}{\mathrm{d}Ma_\infty}\left(
\left.\frac{ℒc_\text{l}}{ℒ\alpha}\right\vert_{s=0}
\right) =
-2 Ma_\infty^{-1}\cdot\left.\frac{ℒc_\text{l}}{ℒ\alpha}\right\vert_{s=0} +
\frac{\partial c_\text{l}}{\partial𝓕}
\frac{\mathrm{d}}{\mathrm{d}Ma_\infty}\left(
\left.\frac{ℒ𝓕}{ℒ\alpha}\right\vert_{s=0}
\right)
\text{,}
$$

with

$$
\frac{\mathrm{d}}{\mathrm{d}Ma_\infty}\left(
\left.\frac{ℒ𝓕}{ℒ\alpha}\right\vert_{s=0}
\right) =
\frac{\partial^2𝓕}{\partial𝓤^2}
\left.\frac{ℒ𝓤}{ℒ\alpha}\right\vert_{s=0}
\left.\frac{ℒ𝓤}{ℒMa_\infty}\right\vert_{s=0} +
\frac{\partial^2𝓕}{\partial{𝓧}\partial𝓤}
\left.\frac{ℒ𝓤}{ℒMa_\infty}\right\vert_{s=0}
\frac{\mathrm{d}𝓧}{\mathrm{d}\alpha} +
\frac{\partial𝓕}{\partial𝓤}
\frac{\mathrm{d}}{\mathrm{d}Ma_\infty}\left(
\left.\frac{ℒ𝓤}{ℒ\alpha}\right\vert_{s=0}
\right)
$$

and

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}Ma_\infty}\left(
\left.\frac{ℒ𝓤}{ℒ\alpha}\right\vert_{s=0}
\right) =
&-\left(\frac{\partial𝓡}{\partial𝓤}\right)^{-1}
\left(
\frac{\partial^2𝓡}{\partial𝓤\partial Ma_\infty} +
\frac{\partial^2𝓡}{\partial𝓤^2}
\left.\frac{ℒ𝓤}{ℒMa_\infty}\right\vert_{s=0}
\right)
\left.\frac{ℒ𝓤}{ℒ\alpha}\right\vert_{s=0}\\
&-\left(\frac{\partial𝓡}{\partial𝓤}\right)^{-1}
\left(
\frac{\partial^2𝓡}{\partial𝓧\partial Ma_\infty} +
\frac{\partial^2𝓡}{\partial𝓧\partial𝓤}
\left.\frac{ℒ𝓤}{ℒMa_\infty}\right\vert_{s=0}
\right)
\frac{\mathrm{d}𝓧}{\mathrm{d}\alpha}\text{.}
\end{aligned}
$$

Simplifying and rearranging yields

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}Ma_\infty}\left(
\left.\frac{ℒc_\text{l}}{ℒ\alpha}\right\vert_{s=0}
\right) =
&-2 Ma_\infty^{-1}\cdot\left.\frac{ℒc_\text{l}}{ℒ\alpha}\right\vert_{s=0}
\\
&+
\left(
\frac{\partial}{\partial𝓤}
\left\langle
\left( \frac{\partial c_\text{l}}{\partial𝓕} \right)^\dagger,
\frac{\partial𝓕}{\partial𝓤}
\left.\frac{ℒ𝓤}{ℒ\alpha}\right\vert_{s=0}
\right\rangle +
\frac{\partial}{\partial𝓤}
\left\langle
\left( \frac{\partial c_\text{l}}{\partial𝓕} \right)^\dagger,
\frac{\partial𝓕}{\partial𝓧}
\frac{\mathrm{d}𝓧}{\mathrm{d}\alpha}
\right\rangle
\right)
\left.\frac{ℒ𝓤}{ℒMa_\infty}\right\vert_{s=0}
\\
&+
\left(
\frac{\partial}{\partial𝓤}
\left\langle
\check{𝓡},
\frac{\partial𝓡}{\partial𝓤}
\left.\frac{ℒ𝓤}{ℒ\alpha}\right\vert_{s=0}
\right\rangle +
\frac{\partial}{\partial𝓤}
\left\langle
\check{𝓡},
\frac{\partial𝓡}{\partial𝓧}
\frac{\mathrm{d}𝓧}{\mathrm{d}\alpha}
\right\rangle
\right)
\left.\frac{ℒ𝓤}{ℒMa_\infty}\right\vert_{s=0}
\\
&+
\frac{\partial}{\partial Ma_\infty}
\left\langle
\check{𝓡},
\frac{\partial𝓡}{\partial𝓤}
\left.\frac{ℒ𝓤}{ℒ\alpha}\right\vert_{s=0}
\right\rangle +
\frac{\partial}{\partial Ma_\infty}
\left\langle
\check{𝓡},
\frac{\partial𝓡}{\partial𝓧}
\left.\frac{ℒ𝓧}{ℒ\alpha}\right\vert_{s=0}
\right\rangle
\end{aligned}
$$

where $\check{𝓡}:=
-(\partial𝓡\textfractionsolidus\partial𝓤)^{-\dagger}
(\partial𝓕\textfractionsolidus\partial𝓤)^\dagger
(\partial c_\text{l}\textfractionsolidus\partial𝓕)^\dagger
$.

Running

```commandline
python examples/gain/run.py examples/grids/naca0004_129x129.plot3d 10. 0.5
```

computes the lift coefficient gain of the NACA-0012, with its chord measuring $`10.0`$ grid units,
in free-stream Mach number $`Ma_\infty=0.5`$, and the derivative of the gain with respect to the
Mach number. The computed gains and derivatives agree with the Prandtl-Glauert rule.


<p align="center">
  <img src=examples/gain/prandtl.png>
</p>

## Copyright

Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved
