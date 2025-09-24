! Copyright (C) 2025 Simon Ehrmanntraut - All Rights Reserved

! Tested for GNU Fortran 11.4.0 with '-O3 -std=f2008' ('-Ofast' is NOT stable)

module spatial_discretization

  !> @brief FVM discretization for the compressible 2D Euler equations around a moving airfoil.
  !!
  !! The Euler equation for a moving cell in two dimensions read
  !!
  !! `d/dt ∫ [ρ; 𝐣; E] dA(t) = - ∮ [𝐣; 𝐣⊗𝐣/ρ+𝟙p; (E+p)⋅𝐣/ϱ] - [ρ⋅𝐯; 𝐣⊗𝐯; E⋅𝐯] d𝐬(t)`
  !!
  !! for the conserved variables (collectively called 'state' `𝓊`) density `ρ`, momentum
  !! densities `𝐣`, and total energy density `E = p/(γ-1) + (𝐣∘𝐣)/ρ/2`. The module implements a
  !! Riemann-solver-free central scheme with Rusanov/Lax-Friedrich flux without limiter yielding
  !! the ordinary differential equation
  !!
  !! ``d𝓤/dt = 𝓡(𝓤,𝓧,𝓥)``
  !!
  !! for the grid cell averaged states `𝓤`, given the grid vertex coordinates `𝓧` and grid vertex
  !! velocities `𝓥`. Only closed O-type grids can be processed; for example, if 'unrolled', a
  !! grid with `N` cells around the airfoil and `M` layers would look like this:
  !!
  !!                           far-field
  !!      ┈┬───────────┬───────────┬───────────┬───────────┬┈┈
  !!       │(M+1,1)    │(M+1,2)    │(M+1,3)    │(M+1,N)    │(M+1,N+1)
  !!   P   │    ●(M,1) │    ●(M,2) │    ●(M,3) │    ●(M,N) │         P
  !!   E   │           │           │           │           │         E
  !!   R  ┈┼───────────┼───────────┼───────────┼───────────┼┈┈       R
  !!   I   │(M,1)      │(M,2)      │(M,3)      │(M,N)      │(M,N+1)  I
  !!   O   │    ●(2,1) │    ●(2,2) │    ●(2,3) │    ●(2,N) │         O
  !!   D   │           │           │           │           │         D
  !!   I  ┈┼───────────┼───────────┼───────────┼───────────┼┈┈       I
  !!   C   │(2,1)      │(2,2)      │(2,3)      │(2,N)      │(2,N+1)  C
  !!       │    ●(1,1) │    ●(1,2) │    ●(1,3) │    ●(1,N) │
  !!       │           │           │           │           │
  !!      ┈┴───────────┴───────────┴───────────┴───────────┼┈┈
  !!     (1,1)       (1,2)       (1,3)       (1,N)       (1,N+1)
  !!                          airfoil surface
  !!
  !! Ensurance of the geometric conservation law is not implemented so the grid movement must not
  !! change the cells' area in time. For non-dimensionalization, free-stream pressure `pₒₒ` and the
  !! free-strem density `ρₒₒ` are taken as reference.
  !!
  !! The module implements the linearization of `𝓡` per the jacobians `∂𝓡/∂𝓤`, `∂𝓡/∂𝓧`, and
  !! `∂𝓡/∂𝓥`, computed through complex-step.

  implicit none

  private

  real(8), parameter :: step = 1e-12_8           !>@brief Complex-step step-size
  complex(8), parameter :: i_step = (0, step)    !>@brief Step-size times imaginary unit
  real(8), parameter :: deg_to_rad = 0.0174532925199432957692369_8  !>@brief Factor π/180
  real(8), parameter :: heat_ratio = 1.4_8       !>@brief Specific heat ratio `γ`
  integer, parameter :: num_dim = 2              !>@brief Number of spatial dimensions
  integer, parameter :: num_var = 4              !>@brief Number of conserved variables per state
  integer, parameter :: num_ode_wrt_state = 5    !>@brief Number of states, a cell's ode depend on
  integer, parameter :: num_ode_wrt_vertex = 4   !>@brief Number of vertices, a cell's ode depend on
  integer, parameter :: num_force_wrt_vertex = 2

  ! To be set through python before calling the procedures
  real(8), public :: mach      !>@brief Free-stream mach number `Maₒₒ`
  real(8), public :: aoa       !>@brief Free-stream angle-of-attack in degrees `α`
  real(8), public :: rusanov   !>@brief Rusanov/Lax-Friedrichs flux factor `cᵣ`

  public set_free_stream_state
  public compute_odes
  public compute_odes_wrt_states
  public convert_odes_wrt_states
  public apply_odes_wrt_states_fwd
  public apply_odes_wrt_states_rev
  public compute_odes_wrt_vertices
  public apply_odes_wrt_vertices_fwd
  public apply_odes_wrt_vertices_rev
  public compute_odes_wrt_velocities
  public compute_forces
  public compute_forces_wrt_states
  public apply_forces_wrt_states_fwd
  public apply_forces_wrt_states_rev
  public compute_forces_wrt_vertices
  public apply_forces_wrt_vertices_fwd
  public apply_forces_wrt_vertices_rev

contains

  !> @brief Complex absolute value function for complex-step.
  !!
  !! Computes `abs_c(a+bi) := abs(a)+abs'(a)⋅bi = abs(a)+sign(a)⋅bi`.
  !!
  !! @param[in] x Complex number
  !! @return Complex absolute value
  pure complex(8) function abs_c(x)
    complex(8), intent(in) :: x
    associate (re => real(x), im => aimag(x))
      abs_c = cmplx(abs(re), im * sign(1._8, re), 8)
    end associate
  end function abs_c

  !> @brief Complex maximum function for complex-step.
  !!
  !! Returns the complex number with the larger real part.
  !!
  !! @param[in] x First complex number
  !! @param[in] y Second Complex number
  !! @return Complex maximum
  pure complex(8) function max_c(x, y)
    complex(8), intent(in) :: x, y
    ! todo
    ! add a small tolerance for improved consistency with real finite-differencing
    if (real(x) > real(y) + 1e-8_8) then
      max_c = x
    else
      max_c = y
    end if
  end function max_c

  !> @brief Pressure from state
  !!
  !! The pressure is `p = (γ-1) ⋅ (E - (𝐣∘𝐣)/ϱ/2)`
  !!
  !! @param[in] state State
  !! @return Pressure
  pure function get_pressure(state) result (pressure)
    complex(8), intent(in) :: state(num_var)
    complex(8) :: pressure
    pressure = (heat_ratio - 1) * &
      (state(4) - sum(state(2:3) * state(2:3)) / state(1) / 2)
  end function get_pressure

  !> @brief Integral of the wave speed across an edge
  !!
  !! The wave speed is `λ = √(γp/ϱ) + |𝐣/ϱ∘𝐧/‖𝐧‖|`; the integral is `∫ λ ds = λ⋅‖𝐧‖`.
  !!
  !! @param[in] state State on the edge
  !! @param[in] normal Normal whose norm is the edge length
  !! @return Wave speed integral
  pure function get_wave_integral(state, normal) result (wave_integral)
    complex(8), intent(in) :: state(num_var)
    complex(8), intent(in) :: normal(num_dim)
    complex(8) :: wave_integral
    wave_integral = abs_c(sum(normal * state(2:3)) / state(1)) + &
      sqrt(heat_ratio * get_pressure(state) / state(1)) * sqrt(sum(normal**2))
  end function get_wave_integral

  !> @brief Physical flux across a stationary edge
  !!
  !! The density flux is `∫ 𝐣∘d𝐬 = ϱ⋅(𝐣/ϱ∘𝐧)`;
  !! the momentum fluxes are `∫ (𝐣⋅𝐣/ϱ + p𝟙)∘d𝐬 = 𝐣⋅(𝐣/ϱ∘𝐧) + p⋅𝐧`;
  !! the energy flux is `∫ (E+p)/ϱ⋅𝐣∘d𝐬 = (E+p)⋅(𝐣/ϱ∘𝐧)`.
  !!
  !! @param[in] state State on the edge
  !! @param[in] normal Normal whose norm is the edge length
  !! @return Physical flux
  pure function make_physical_flux(state, normal) result (flux)
    complex(8), intent(in) :: state(num_var)
    complex(8), intent(in) :: normal(num_dim)
    complex(8) :: flux(num_var)
    associate (&
      velocity_integral => sum(normal * state(2:3)) / state(1), &
      pressure => get_pressure(state)&
      )
      flux(1) = state(1) * velocity_integral
      flux(2:3) = state(2:3) * velocity_integral + normal * get_pressure(state)
      flux(4) = (state(4) + get_pressure(state)) * velocity_integral
    end associate
  end function make_physical_flux

  !> @brief Numerical flux across a moving edge
  !!
  !! The numerical flux is `𝒻 = (𝒻ᶜ(𝓊₁)+𝒻ᶜ(𝓊₂))/2 + cᵣ(𝓊₁-𝓊₂)/2⋅max(λ(𝓊₁),λ(𝓊₂)) + (𝓊₁+𝓊₂)/2⋅𝐯∘𝐧`,
  !! where `𝓊₁` and `𝓊₂` are the states in cell 1 and cell 2, `𝒻ᶜ` is the physical flux function,
  !! `𝐧` is the normal vector pointing from cell 1 to cell 2, `𝐯` is the grid velocity, `λ` is the
  !! wave speed integral function, and `cᵣ` is the Rusanov/Lax-Friedrichs flux factor.
  !!
  !! @param[in] state_1 State in cell 1
  !! @param[in] state_2 State in cell 2
  !! @param[in] normal Normal pointing from cell 1 to cell 2 whose norm is the edge length
  !! @param[in] grid_velo Grid velocity on the edge center
  !! @return Numerical flux
  pure function make_face_flux(state_1, state_2, normal, grid_velo) result (flux)
    complex(8), intent(in) :: state_1(num_var)
    complex(8), intent(in) :: state_2(num_var)
    complex(8), intent(in) :: normal(num_dim)
    complex(8), intent(in) :: grid_velo(num_dim)
    complex(8) :: flux(num_var)
    flux = (make_physical_flux(state_1, normal) + make_physical_flux(state_2, normal)) / 2
    ! ALE
    flux = flux - (state_1 + state_2) / 2 * sum(grid_velo * normal)
    ! Rusanov
    associate (&
      wave_integral_1 => get_wave_integral(state_1, normal), &
      wave_integral_2 => get_wave_integral(state_2, normal) &
      )
      flux = flux + (state_1 - state_2) / 2 * rusanov * max_c(wave_integral_1, wave_integral_2)
    end associate
  end function make_face_flux

  !> @brief Ghost state for the wall boundary
  !!
  !! The average of the ghost state ('inside' the airfoil) and the wall state (having the
  !! airfoil as edge) is to have its velocity parallel to the airfoil's tangent.
  !!
  !! @param[in] wall_state State in the wall cell
  !! @param[in] wall_normal Normal pointing into the wall
  !! @param[in] wall_grid_velo Grid velocity on the edge center
  !! @return Ghost state
  pure function make_wall_ghost_state(wall_state, wall_normal, wall_grid_velo) result (ghost_state)
    complex(8), intent(in) :: wall_state(num_var)
    complex(8), intent(in) :: wall_normal(num_dim)
    complex(8), intent(in) :: wall_grid_velo(num_dim)
    complex(8) :: ghost_state(num_var)
    ghost_state(1) = wall_state(1)
    ghost_state(2:3) = wall_state(2:3) - 2 * wall_normal * &
      sum(wall_normal * (wall_state(2:3) - wall_grid_velo * wall_state(1))) / sum(wall_normal**2)
    ghost_state(4) = wall_state(4)
  end function make_wall_ghost_state

  !> @brief Free-stream state
  !!
  !! The non-dimensional density is `ϱₒₒ/ϱₒₒ = 1`;
  !! The non-dimensional speed is `aₒₒMaₒₒ/√(pₒₒ/ϱₒₒ) = Maₒₒ√γ`;
  !! The non-dimensional total energy is `Eₒₒ/pₒₒ = (pₒₒ/(γ-1) + ϱₒₒaₒₒ²Maₒₒ²/2)/pₒₒ = 1/(γ-1) + γMaₒₒ²/2`.
  !!
  !! @return Free stream state
  pure function make_free_stream_state() result (state)
    complex(8) :: state(num_var)
    associate (free_stream_speed => mach * sqrt(heat_ratio), aoa_rad => deg_to_rad * aoa)
      state(1) = 1
      state(2) = free_stream_speed * cos(aoa_rad)
      state(3) = free_stream_speed * sin(aoa_rad)
      state(4) = 1 / (heat_ratio - 1) + free_stream_speed**2 / 2
    end associate
  end function make_free_stream_state

  !> @brief Ghost state for the far-field boundary
  !!
  !! The average of the ghost state ('outside' the domain) and the far-field state (having the
  !! domain boundary as edge) is to match the free-stream state.
  !!
  !! @param[in] farfield_state State in the far-field cell
  !! @return Ghost state
  pure function make_farfield_ghost_state(farfield_state) result (ghost_state)
    complex(8), intent(in) :: farfield_state(num_var)
    complex(8) :: ghost_state(num_var)
    ghost_state = 2 * make_free_stream_state() - farfield_state
  end function make_farfield_ghost_state

  !> @brief Cell area
  !!
  !! BO┌───┐ FO
  !!   │   │
  !! BI└───┘ FI
  !!
  !! @param[in] vertex_bi Cell vertex coordinates 'backward-inward'
  !! @param[in] vertex_fi Cell Vertex coordinates 'forward-inward'
  !! @param[in] vertex_fo Cell Vertex coordinates 'forward-outward'
  !! @param[in] vertex_bo Cell Vertex coordinates 'backward-outward'
  !! @return Area
  pure function make_area(vertex_bi, vertex_fi, vertex_fo, vertex_bo) result (area)
    complex(8), dimension(num_dim), intent(in) :: vertex_bi, vertex_fi, vertex_fo, vertex_bo
    complex(8) :: area
    associate (diag_1 => vertex_fo - vertex_bi, diag_2 => vertex_bo - vertex_fi)
      area = abs_c(diag_1(1) * diag_2(2) - diag_1(2) * diag_2(1)) / 2
    end associate
  end function make_area

  !> @brief Edge Normal
  !!
  !! z     n
  !! │     │
  !! │   ←─┴──
  !! │  V2   V1
  !! └────────x
  !!
  !! @param[in] vertex_1 Vertex coordinates 'from'
  !! @param[in] vertex_2 Vertex coordinates 'to'
  !! @return Area
  pure function make_normal(vertex_1, vertex_2) result(normal)
    complex(8), dimension(num_dim), intent(in) :: vertex_1, vertex_2
    complex(8) :: normal(num_dim)
    normal(1) = vertex_2(2) - vertex_1(2)
    normal(2) = vertex_1(1) - vertex_2(1)
  end function make_normal

  !> @brief Grid velocity on edge
  !!
  !! Simply the average of the grid velocities on the vertices.
  !!
  !! @param[in] velo_1 Vertex velocity on first edge vertex
  !! @param[in] velo_2 Vertex velocity on second edge vertex
  !! @return Grid velocity on edge
  pure function make_grid_velo(velo_1, velo_2) result(grid_velo)
    complex(8), dimension(num_dim), intent(in) :: velo_1, velo_2
    complex(8) :: grid_velo(num_dim)
    grid_velo = (velo_1 + velo_2) / 2
  end function make_grid_velo

  !> @brief Cell's state rate-of-change
  !!
  !! The rate of change `d𝓊ᶜ/dt = 𝓇(𝓊ᶜ,𝓊ᵇ,𝓊ᶠ,𝓊ⁱ,𝓊ᵒ,𝓍ᶜ,𝓋ᶜ)` is the negative sum of all the edge fluxes
  !! over the cell area, i.e. `𝓇 = -1/aᶜ ⋅ [𝒻(𝓊ᶜ,𝓊ᵇ,𝐧ᵇ,𝐯ᵇ)+𝒻(𝓊ᶜ,𝓊ᶠ,𝐧ᶠ,𝐯ᶠ)+𝒻(𝓊ᶜ,𝓊ⁱ,𝐧ⁱ,𝐯ⁱ)+𝒻(𝓊ᶜ,𝓊ᵒ,𝐧ᵒ,𝐯ᵒ)]`.
  !! If the boundary flag is 1 the 'outward' state is ignored and the far-field ghost state is used
  !! instead; if the flat is -1 the 'inward' state is ignored and the wall ghost state is used
  !! instead.
  !!
  !!     ┌───┐
  !!  BO │ O │ FO
  !! ┌───┼───┼───┐
  !! │ B │ C │ F │
  !! └───┼───┼───┘
  !!  BO │ I │ FO
  !!     └───┘
  !!
  !! @param[in] vertex_bi Cell vertex coordinates 'backward-inward'
  !! @param[in] vertex_fi Cell Vertex coordinates 'forward-inward'
  !! @param[in] vertex_fo Cell Vertex coordinates 'forward-outward'
  !! @param[in] vertex_bo Cell Vertex coordinates 'backward-outward'
  !! @param[in] velo_bi Cell vertex velocity 'backward-inward'
  !! @param[in] velo_fi Cell vertex velocity 'forward-inward'
  !! @param[in] velo_fo Cell vertex velocity 'forward-outward'
  !! @param[in] velo_bo Cell vertex velocityy 'backward-outward'
  !! @param[in] state_c Cell state 'center'
  !! @param[in] state_b Cell state 'backward'
  !! @param[in] state_f Cell state 'forward'
  !! @param[in] state_i Cell state 'inward'
  !! @param[in] state_o Cell state 'outward'
  !! @param[in] boundary_flag
  !! @return Rate of change of cell state 'center'
  pure function make_ode(&
    vertex_bi, vertex_fi, vertex_fo, vertex_bo, velo_bi, velo_fi, velo_fo, velo_bo, &
    state_c, state_b, state_f, state_i, state_o, boundary_flag&
    ) result (ode)
    complex(8), dimension(num_dim), intent(in) :: vertex_bi, vertex_fi, vertex_fo, vertex_bo
    complex(8), dimension(num_dim), intent(in) :: velo_bi, velo_fi, velo_fo, velo_bo
    complex(8), dimension(num_var), intent(in) :: state_c, state_b, state_f
    complex(8), dimension(num_var), optional, intent(in) :: state_i, state_o
    integer, intent(in) :: boundary_flag
    complex(8) :: ode(num_var)
    associate (&
      area => make_area(vertex_bi, vertex_fi, vertex_fo, vertex_bo), &
      normal_i => make_normal(vertex_bi, vertex_fi), &
      normal_f => make_normal(vertex_fi, vertex_fo), &
      normal_o => make_normal(vertex_fo, vertex_bo), &
      normal_b => make_normal(vertex_bo, vertex_bi), &
      grid_velo_i => make_grid_velo(velo_bi, velo_fi), &
      grid_velo_f => make_grid_velo(velo_fi, velo_fo), &
      grid_velo_o => make_grid_velo(velo_fo, velo_bo), &
      grid_velo_b => make_grid_velo(velo_bo, velo_bi) &
      )
      select case (boundary_flag)
      case default
        ode = -(make_face_flux(state_c, state_b, normal_b, grid_velo_b) + &
          make_face_flux(state_c, state_f, normal_f, grid_velo_f) + &
          make_face_flux(state_c, state_i, normal_i, grid_velo_i) + &
          make_face_flux(state_c, state_o, normal_o, grid_velo_o)&
          ) / area
      case (-1)
        ode = -(make_face_flux(state_c, state_b, normal_b, grid_velo_b) + &
          make_face_flux(state_c, state_f, normal_f, grid_velo_f) + &
          make_face_flux(state_c, make_wall_ghost_state(state_c, normal_i, grid_velo_i), &
            normal_i, grid_velo_i) + &
          make_face_flux(state_c, state_o, normal_o, grid_velo_o)&
          ) / area
      case (1)
        ode = -(make_face_flux(state_c, state_b, normal_b, grid_velo_b) + &
          make_face_flux(state_c, state_f, normal_f, grid_velo_f) + &
          make_face_flux(state_c, state_i, normal_i, grid_velo_i) + &
          make_face_flux(state_c, make_farfield_ghost_state(state_c), normal_o, grid_velo_o)&
          ) / area
      end select
    end associate
  end function make_ode

  ! =====================

  !> @brief Boundary flag
  !!
  !! If the radial index equals 1 the cell is at the airfoil's wall, the boundary flag is -1. If
  !! the radial index equals the number of cells in the radial direction the, the boundary flag
  !! is 1. In between, the boundary flag is 0.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] m Radial index
  !! @return Boundary flag
  pure function get_boundary_flag(num_radial, m) result (boundary_flag)
    integer, intent(in) :: num_radial, m
    integer :: boundary_flag
    if (m  == 1) then
      boundary_flag = -1
    else if (m == num_radial) then
      boundary_flag = 1
    else
      boundary_flag = 0
    end if
  end function get_boundary_flag

  !> @brief Sets states to free-stream
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[inout] states States.
  pure subroutine set_free_stream_state(num_radial, num_angular, states)
    !f2py integer, intent(hide), depend(states) :: num_radial = size(states, 2)
    !f2py integer, intent(hide), depend(states) :: num_angular = size(states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2
    integer, intent(in) :: num_radial, num_angular
    complex(8), intent(inout) :: states(num_var, num_radial, num_angular)
    integer :: m, n
    do concurrent (n = 1:num_angular, m = 1:num_radial)
      states(:, m, n) = make_free_stream_state()
    end do
  end subroutine set_free_stream_state

  !> @brief Computes states' rate-of-change
  !!
  !! The states' rate-of-change are `d𝓤/dt = 𝓡(𝓤,𝓧,𝓥)`, where `𝓤` are the
  !! states, `𝓧` are the grid vertex coordinates, and `𝓥` are the grid vertex velocities.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] vertices Grid vertex coordinate
  !! @param[in] velocities Grid vertex velocities
  !! @param[in] states States
  !! @param[inout] states States' rate-of-change
  pure subroutine compute_odes(&
    num_radial, num_angular, vertices, velocities, states, odes)
    !f2py integer, intent(hide), depend(states) :: num_radial = size(states, 2)
    !f2py integer, intent(hide), depend(states) :: num_angular = size(states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2
    integer, intent(in) :: num_radial, num_angular
    complex(8), intent(in) :: vertices(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: velocities(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: states(num_var, num_radial, num_angular)
    complex(8), intent(inout) :: odes(num_var, num_radial, num_angular)
    integer :: m, n
    odes = 0
    do concurrent (n = 1:num_angular, m = 1:num_radial)
      associate (&
        vertex_bi => vertices(:, m, n), &
        vertex_fi => vertices(:, m, n + 1), &
        vertex_fo => vertices(:, m + 1, n + 1), &
        vertex_bo => vertices(:, m + 1, n), &
        velo_bi => velocities(:, m, n), &
        velo_fi => velocities(:, m, n + 1), &
        velo_fo => velocities(:, m + 1, n + 1), &
        velo_bo => velocities(:, m + 1, n), &
        state_c => states(:, m, n), &
        state_b => states(:, m, modulo(n - 2, num_angular) + 1), &
        state_f => states(:, m, modulo(n, num_angular) + 1), &
        state_i => states(:, m - 1, n), &
        state_o => states(:, m + 1, n), &
        boundary_flag => get_boundary_flag(num_radial, m), &
        ode => odes(:, m, n)&
        )
        ode = make_ode(vertex_bi, vertex_fi, vertex_fo, vertex_bo, &
          velo_bi, velo_fi, velo_fo, velo_bo, &
          state_c, state_b, state_f, state_i, state_o, boundary_flag)
      end associate
    end do
  end subroutine compute_odes

  !> @brief Computes jacobians of ``compute_odes`` with respect to ``states``
  !!
  !! The jacobians `∂𝓡/∂𝓤` are to be used in
  !! * apply_odes_wrt_states_fwd,
  !! * apply_odes_wrt_states_rev, and
  !! * convert_odes_wrt_states.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] vertices Grid vertex coordinate
  !! @param[in] velocities Grid vertex velocities
  !! @param[in] states States
  !! @param[inout] jacs Jacobians
  pure subroutine compute_odes_wrt_states(&
    num_radial, num_angular, vertices, velocities, states, jacs)
    !f2py integer, intent(hide), depend(states) :: num_radial = size(states, 2)
    !f2py integer, intent(hide), depend(states) :: num_angular = size(states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_ode_wrt_state = 5
    integer, intent(in) :: num_radial, num_angular
    complex(8), intent(in) :: vertices(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: velocities(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: states(num_var, num_radial, num_angular)
    real(8), intent(inout) :: jacs(num_var, num_var, num_radial, num_angular, num_ode_wrt_state)
    complex(8) :: d_state(num_var)
    integer :: i, m, n
    do concurrent (n = 1:num_angular, m = 1:num_radial)
      associate (&
        vertex_bi => vertices(:, m, n), &
        vertex_fi => vertices(:, m, n + 1), &
        vertex_fo => vertices(:, m + 1, n + 1), &
        vertex_bo => vertices(:, m + 1, n), &
        velo_bi => velocities(:, m, n), &
        velo_fi => velocities(:, m, n + 1), &
        velo_fo => velocities(:, m + 1, n + 1), &
        velo_bo => velocities(:, m + 1, n), &
        state_c => states(:, m, n), &
        state_b => states(:, m, modulo(n - 2, num_angular) + 1), &
        state_f => states(:, m, modulo(n, num_angular) + 1), &
        state_i => states(:, m - 1, n), &
        state_o => states(:, m + 1, n), &
        boundary_flag => get_boundary_flag(num_radial, m), &
        jac_b => jacs(:, :, m, n, 1), &
        jac_i => jacs(:, :, m, n, 2), &
        jac_c => jacs(:, :, m, n, 3), &
        jac_o => jacs(:, :, m, n, 4), &
        jac_f => jacs(:, :, m, n, 5) &
        )
        do concurrent (i = 1:num_var)
          d_state = 0
          d_state(i) = i_step
          jac_c(:, i) = aimag(make_ode(&
            vertex_bi, vertex_fi, vertex_fo, vertex_bo, velo_bi, velo_fi, velo_fo, velo_bo, &
            state_c + d_state, state_b, state_f, state_i, state_o, boundary_flag)) / step
          jac_b(:, i) = aimag(make_ode(&
            vertex_bi, vertex_fi, vertex_fo, vertex_bo, velo_bi, velo_fi, velo_fo, velo_bo, &
            state_c, state_b + d_state, state_f, state_i, state_o, boundary_flag)) / step
          jac_f(:, i) = aimag(make_ode(&
            vertex_bi, vertex_fi, vertex_fo, vertex_bo, velo_bi, velo_fi, velo_fo, velo_bo, &
            state_c, state_b, state_f + d_state, state_i, state_o, boundary_flag)) / step
          if (m > 1) jac_i(:, i) = aimag(make_ode(&
            vertex_bi, vertex_fi, vertex_fo, vertex_bo, velo_bi, velo_fi, velo_fo, velo_bo, &
            state_c, state_b, state_f, state_i + d_state, state_o, boundary_flag)) / step
          if (m < num_radial) jac_o(:, i) = aimag(make_ode(&
            vertex_bi, vertex_fi, vertex_fo, vertex_bo, velo_bi, velo_fi, velo_fo, velo_bo, &
            state_c, state_b, state_f, state_i, state_o + d_state, boundary_flag)) / step
        end do
      end associate
    end do
  end subroutine compute_odes_wrt_states

  !> @brief Converts the jacobians of ``compute_odes`` with respect to ``states`` into Block
  !! Sparse Row (BSR) format.
  !!
  !! In block row ``i`` the, column indices are ``indices(index_pointers(i):index_pointers(i+1)-1)``
  !! and the corresponding 4x4 blocks are ``data(index_pointers(i):index_pointers(i+1)-1)``.
  !! See SciPy's ``scipy.sparse.bsr_array`` for a related format.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] jacs Jacobians
  !! @param[inout] data BSR format data
  !! @param[inout] indices BSR format indices
  !! @param[inout] index_pointers BSR format index pointers
  subroutine convert_odes_wrt_states(&
    num_radial, num_angular, jacs, data, indices, index_pointers)
    !f2py integer, intent(hide), depend(jacs) :: num_radial = size(jacs, 3)
    !f2py integer, intent(hide), depend(jacs) :: num_angular = size(jacs, 4)
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_ode_wrt_state = 5
    integer, intent(in) :: num_radial, num_angular
    real(8), intent(in) :: jacs(num_var, num_var, num_radial, num_angular, num_ode_wrt_state)
    real(8), intent(inout) :: data(num_var, num_var, (num_radial * 5 - 2) * num_angular)
    integer(4), intent(inout) :: indices((num_radial * 5 - 2) * num_angular)
    integer(4), intent(inout) :: index_pointers(num_radial * num_angular + 1)
    integer :: n_row, m_row, n_col, m_col, row, col, ptr
    index_pointers(1) = 1
    ptr = 1
    row = 0
    do n_row = 1, num_angular
      do m_row = 1, num_radial
        row = row + 1
        col = 0
        do n_col = 1, num_angular
          do m_col = 1, num_radial
            col = col + 1
            if (m_col == m_row .and. modulo(n_col - n_row + 1, num_angular) == 0) then
              data(:, :, ptr) = jacs(:, :, m_row, n_row, 1)
            else if (m_col == m_row - 1 .and. n_col == n_row) then
              data(:, :, ptr) = jacs(:, :, m_row, n_row, 2)
            else if (m_col == m_row .and. n_col == n_row) then
              data(:, :, ptr) = jacs(:, :, m_row, n_row, 3)
            else if (m_col == m_row + 1 .and. n_col == n_row) then
              data(:, :, ptr) = jacs(:, :, m_row, n_row, 4)
            else if (m_col == m_row .and. modulo(n_col - n_row - 1, num_angular) == 0) then
              data(:, :, ptr) = jacs(:, :, m_row, n_row, 5)
            else
              cycle
            end if
            indices(ptr) = col
            ptr = ptr + 1
          end do
        end do
        index_pointers(row + 1) = ptr
      end do
    end do
  end subroutine convert_odes_wrt_states

  !> @brief Applies jacobians of ``compute_odes`` with respect to ``states`` in forward mode
  !!
  !! Computes the matrix-vector-product `∂𝓡/∂𝓤⋅δ𝓤`, i.e. the directional derivative.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] jacs Jacobians from ``compute_odes_wrt_states``
  !! @param[in] d_states Vector to multiply to the Jacobians
  !! @param[inout] d_odes Vector-product
  pure subroutine apply_odes_wrt_states_fwd(num_radial, num_angular, jacs, d_states, d_odes)
    !f2py integer, intent(hide), depend(d_odes) :: num_radial = size(d_odes, 2)
    !f2py integer, intent(hide), depend(d_odes) :: num_angular = size(d_odes, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_ode_wrt_state = 5
    integer, intent(in) :: num_radial, num_angular
    real(8), intent(in) :: jacs(num_var, num_var, num_radial, num_angular, num_ode_wrt_state)
    complex(8), intent(in) :: d_states(num_var, num_radial, num_angular)
    complex(8), intent(inout) :: d_odes(num_var, num_radial, num_angular)
    integer :: m, n
    do concurrent (n = 1:num_angular, m = 1:num_radial)
      associate (&
        d_state_c => d_states(:, m, n), &
        d_state_b => d_states(:, m, modulo(n - 2, num_angular) + 1), &
        d_state_f => d_states(:, m, modulo(n, num_angular) + 1), &
        d_state_i => d_states(:, m - 1, n), &
        d_state_o => d_states(:, m + 1, n), &
        jac_b => jacs(:, :, m, n, 1), &
        jac_i => jacs(:, :, m, n, 2), &
        jac_c => jacs(:, :, m, n, 3), &
        jac_o => jacs(:, :, m, n, 4), &
        jac_f => jacs(:, :, m, n, 5), &
        d_ode => d_odes(:, m, n)&
        )
        d_ode = matmul(jac_c, d_state_c) + matmul(jac_b, d_state_b) + matmul(jac_f, d_state_f)
        if (m > 1) d_ode = d_ode + matmul(jac_i, d_state_i)
        if (m < num_radial) d_ode = d_ode + matmul(jac_o, d_state_o)
      end associate
    end do
  end subroutine apply_odes_wrt_states_fwd

  !> @brief Applies jacobians of ``compute_odes`` with respect to ``states`` in reverse mode
  !!
  !! Computes the matrix-vector-product `∂𝓡/∂𝓤ᵀ⋅δ𝓡`.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] jacs Jacobians from ``compute_odes_wrt_states``
  !! @param[in] d_odes Covector to multiply to the Jacobians
  !! @param[inout] d_states Covector-product
  pure subroutine apply_odes_wrt_states_rev(num_radial, num_angular, jacs, d_odes, d_states)
    !f2py integer, intent(hide), depend(d_odes) :: num_radial = size(d_odes, 2)
    !f2py integer, intent(hide), depend(d_odes) :: num_angular = size(d_odes, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_ode_wrt_state = 5
    integer, intent(in) :: num_radial, num_angular
    real(8), intent(in) :: jacs(num_var, num_var, num_radial, num_angular, num_ode_wrt_state)
    complex(8), intent(in) :: d_odes(num_var, num_radial, num_angular)
    complex(8), intent(inout) :: d_states(num_var, num_radial, num_angular)
    integer :: m, n
    d_states = 0
    do concurrent (n = 1:num_angular, m = 1:num_radial)
      associate (&
        d_ode => d_odes(:, m, n), &
        jac_b => jacs(:, :, m, n, 1), &
        jac_i => jacs(:, :, m, n, 2), &
        jac_c => jacs(:, :, m, n, 3), &
        jac_o => jacs(:, :, m, n, 4), &
        jac_f => jacs(:, :, m, n, 5), &
        d_state_c => d_states(:, m, n), &
        d_state_b => d_states(:, m, modulo(n - 2, num_angular) + 1), &
        d_state_f => d_states(:, m, modulo(n, num_angular) + 1), &
        d_state_i => d_states(:, m - 1, n), &
        d_state_o => d_states(:, m + 1, n)&
        )
        d_state_c = d_state_c + matmul(d_ode, jac_c)
        d_state_b = d_state_b + matmul(d_ode, jac_b)
        d_state_f = d_state_f + matmul(d_ode, jac_f)
        if (m > 1) d_state_i = d_state_i + matmul(d_ode, jac_i)
        if (m < num_radial) d_state_o = d_state_o + matmul(d_ode, jac_o)
      end associate
    end do
  end subroutine apply_odes_wrt_states_rev

  !> @brief Computes jacobians of ``compute_odes`` with respect to ``vertices``
  !!
  !! The jacobians `∂𝓡/∂𝓧` are to be used in
  !! * apply_odes_wrt_vertices_fwd, and
  !! * apply_odes_wrt_vertices_rev.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] vertices Grid vertex coordinate
  !! @param[in] velocities Grid vertex velocities
  !! @param[in] states States
  !! @param[inout] jacs Jacobians
  pure subroutine compute_odes_wrt_vertices(&
    num_radial, num_angular, vertices, velocities, states, jacs)
    !f2py integer, intent(hide), depend(states) :: num_radial = size(states, 2)
    !f2py integer, intent(hide), depend(states) :: num_angular = size(states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_ode_wrt_vertex = 4
    integer, intent(in) :: num_radial, num_angular
    complex(8), intent(in) :: vertices(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: velocities(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: states(num_var, num_radial, num_angular)
    real(8), intent(inout) :: jacs(num_var, num_dim, num_radial, num_angular, num_ode_wrt_vertex)
    complex(8) :: d_vertex(num_dim)
    integer :: i, m, n
    do concurrent (n = 1:num_angular, m = 1:num_radial)
      associate (&
        vertex_bi => vertices(:, m, n), &
        vertex_fi => vertices(:, m, n + 1), &
        vertex_fo => vertices(:, m + 1, n + 1), &
        vertex_bo => vertices(:, m + 1, n), &
        velo_bi => velocities(:, m, n), &
        velo_fi => velocities(:, m, n + 1), &
        velo_fo => velocities(:, m + 1, n + 1), &
        velo_bo => velocities(:, m + 1, n), &
        state_c => states(:, m, n), &
        state_b => states(:, m, modulo(n - 2, num_angular) + 1), &
        state_f => states(:, m, modulo(n, num_angular) + 1), &
        state_i => states(:, m - 1, n), &
        state_o => states(:, m + 1, n), &
        boundary_flag => get_boundary_flag(num_radial, m), &
        jac_bi => jacs(:, :, m, n, 1), &
        jac_fi => jacs(:, :, m, n, 2), &
        jac_fo => jacs(:, :, m, n, 3), &
        jac_bo => jacs(:, :, m, n, 4) &
        )
        do concurrent (i = 1:num_dim)
          d_vertex = 0
          d_vertex(i) = i_step
          jac_bi(:, i) = aimag(make_ode(vertex_bi + d_vertex, vertex_fi, vertex_fo, vertex_bo, &
            velo_bi, velo_fi, velo_fo, velo_bo, state_c, state_b, state_f, state_i, state_o, &
            boundary_flag)) / step
          jac_fi(:, i) = aimag(make_ode(vertex_bi, vertex_fi + d_vertex, vertex_fo, vertex_bo, &
            velo_bi, velo_fi, velo_fo, velo_bo, state_c, state_b, state_f, state_i, state_o, &
            boundary_flag)) / step
          jac_fo(:, i) = aimag(make_ode(vertex_bi, vertex_fi, vertex_fo + d_vertex, vertex_bo, &
            velo_bi, velo_fi, velo_fo, velo_bo, state_c, state_b, state_f, state_i, state_o, &
            boundary_flag)) / step
          jac_bo(:, i) = aimag(make_ode(vertex_bi, vertex_fi, vertex_fo, vertex_bo + d_vertex, &
            velo_bi, velo_fi, velo_fo, velo_bo, state_c, state_b, state_f, state_i, state_o, &
            boundary_flag)) / step
        end do
      end associate
    end do
  end subroutine compute_odes_wrt_vertices

  !> @brief Applies jacobians of ``compute_odes`` with respect to ``vertices`` (``velocities``) in
  !! forward mode
  !!
  !! Computes the matrix-vector-product `∂𝓡/∂𝓧⋅δ𝓧` (`∂𝓡/∂𝓥⋅δ𝓥`).
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] jacs Jacobians from ``compute_odes_wrt_vertices`` (``compute_odes_wrt_velocities``)
  !! @param[in] d_vertices Vector to multiply to the Jacobians
  !! @param[inout] d_odes Vector-product
  pure subroutine apply_odes_wrt_vertices_fwd(num_radial, num_angular, jacs, d_vertices, d_odes)
    !f2py integer, intent(hide), depend(d_odes) :: num_radial = size(d_odes, 2)
    !f2py integer, intent(hide), depend(d_odes) :: num_angular = size(d_odes, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_ode_wrt_vertex = 4
    integer, intent(in) :: num_radial, num_angular
    real(8), intent(in) :: jacs(num_var, num_dim, num_radial, num_angular, num_ode_wrt_vertex)
    complex(8), intent(in) :: d_vertices(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(inout) :: d_odes(num_var, num_radial, num_angular)
    integer :: m, n
    do concurrent (n = 1:num_angular, m = 1:num_radial)
      associate (&
        d_vertex_bi => d_vertices(:, m, n), &
        d_vertex_fi => d_vertices(:, m, n + 1), &
        d_vertex_fo => d_vertices(:, m + 1, n + 1), &
        d_vertex_bo => d_vertices(:, m + 1, n), &
        jac_bi => jacs(:, :, m, n, 1), &
        jac_fi => jacs(:, :, m, n, 2), &
        jac_fo => jacs(:, :, m, n, 3), &
        jac_bo => jacs(:, :, m, n, 4), &
        d_ode => d_odes(:, m, n)&
        )
        d_ode = (matmul(jac_bi, d_vertex_bi) + &
          matmul(jac_fi, d_vertex_fi) + &
          matmul(jac_fo, d_vertex_fo) + &
          matmul(jac_bo, d_vertex_bo))
      end associate
    end do
  end subroutine apply_odes_wrt_vertices_fwd

  !> @brief Applies jacobians of ``compute_odes`` with respect to ``vertices`` (``velocities``) in
  !! reverse mode
  !!
  !! Computes the matrix-vector-product `∂𝓡/∂𝓧ᵀ⋅δ𝓡` (`∂𝓡/∂𝓥ᵀ⋅δ𝓡`).
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] jacs Jacobians from ``compute_odes_wrt_vertices`` (``compute_odes_wrt_velocities``)
  !! @param[in] d_odes Covector to multiply to the Jacobians
  !! @param[inout] d_vertices Covector-product
  pure subroutine apply_odes_wrt_vertices_rev(num_radial, num_angular, jacs, d_odes, d_vertices)
    !f2py integer, intent(hide), depend(d_odes) :: num_radial = size(d_odes, 2)
    !f2py integer, intent(hide), depend(d_odes) :: num_angular = size(d_odes, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_ode_wrt_vertex = 4
    integer, intent(in) :: num_radial, num_angular
    real(8), intent(in) :: jacs(num_var, num_dim, num_radial, num_angular, num_ode_wrt_vertex)
    complex(8), intent(in) :: d_odes(num_var, num_radial, num_angular)
    complex(8), intent(inout) :: d_vertices(num_dim, num_radial + 1, num_angular + 1)
    integer :: m, n
    d_vertices = 0
    do concurrent (n = 1:num_angular, m = 1:num_radial)
      associate (&
        d_ode => d_odes(:, m, n), &
        jac_bi => jacs(:, :, m, n, 1), &
        jac_fi => jacs(:, :, m, n, 2), &
        jac_fo => jacs(:, :, m, n, 3), &
        jac_bo => jacs(:, :, m, n, 4), &
        d_vertex_bi => d_vertices(:, m, n), &
        d_vertex_fi => d_vertices(:, m, n + 1), &
        d_vertex_fo => d_vertices(:, m + 1, n + 1), &
        d_vertex_bo => d_vertices(:, m + 1, n)&
        )
        d_vertex_bi = d_vertex_bi + matmul(d_ode, jac_bi)
        d_vertex_fi = d_vertex_fi + matmul(d_ode, jac_fi)
        d_vertex_fo = d_vertex_fo + matmul(d_ode, jac_fo)
        d_vertex_bo = d_vertex_bo + matmul(d_ode, jac_bo)
      end associate
    end do
  end subroutine apply_odes_wrt_vertices_rev

  !> @brief Computes jacobians of ``compute_odes`` with respect to ``velocities``
  !!
  !! The jacobians `∂𝓡/∂𝓥` are to be used (reusing same functionality) in
  !! * apply_odes_wrt_vertices_fwd, and
  !! * apply_odes_wrt_vertices_rev.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] vertices Grid vertex coordinate
  !! @param[in] velocities Grid vertex velocities
  !! @param[in] states States
  !! @param[inout] jacs Jacobians
  pure subroutine compute_odes_wrt_velocities(&
    num_radial, num_angular, vertices, velocities, states, jacs)
    !f2py integer, intent(hide), depend(states) :: num_radial = size(states, 2)
    !f2py integer, intent(hide), depend(states) :: num_angular = size(states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_ode_wrt_vertex = 4
    integer, intent(in) :: num_radial, num_angular
    complex(8), intent(in) :: vertices(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: velocities(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: states(num_var, num_radial, num_angular)
    real(8), intent(inout) :: jacs(num_var, num_dim, num_radial, num_angular, num_ode_wrt_vertex)
    complex(8) :: d_velo(num_dim)
    integer :: i, m, n
    do concurrent (n = 1:num_angular, m = 1:num_radial)
      associate (&
        vertex_bi => vertices(:, m, n), &
        vertex_fi => vertices(:, m, n + 1), &
        vertex_fo => vertices(:, m + 1, n + 1), &
        vertex_bo => vertices(:, m + 1, n), &
        velo_bi => velocities(:, m, n), &
        velo_fi => velocities(:, m, n + 1), &
        velo_fo => velocities(:, m + 1, n + 1), &
        velo_bo => velocities(:, m + 1, n), &
        state_c => states(:, m, n), &
        state_b => states(:, m, modulo(n - 2, num_angular) + 1), &
        state_f => states(:, m, modulo(n, num_angular) + 1), &
        state_i => states(:, m - 1, n), &
        state_o => states(:, m + 1, n), &
        boundary_flag => get_boundary_flag(num_radial, m), &
        jac_bi => jacs(:, :, m, n, 1), &
        jac_fi => jacs(:, :, m, n, 2), &
        jac_fo => jacs(:, :, m, n, 3), &
        jac_bo => jacs(:, :, m, n, 4) &
        )
        do concurrent (i = 1:num_dim)
          d_velo = 0
          d_velo(i) = i_step
          jac_bi(:, i) = aimag(make_ode(vertex_bi, vertex_fi, vertex_fo, vertex_bo, &
            velo_bi + d_velo, velo_fi, velo_fo, velo_bo, &
            state_c, state_b, state_f, state_i, state_o, boundary_flag)) / step
          jac_fi(:, i) = aimag(make_ode(vertex_bi, vertex_fi, vertex_fo, vertex_bo, &
            velo_bi, velo_fi + d_velo, velo_fo, velo_bo, &
            state_c, state_b, state_f, state_i, state_o, boundary_flag)) / step
          jac_fo(:, i) = aimag(make_ode(vertex_bi, vertex_fi, vertex_fo, vertex_bo, &
            velo_bi, velo_fi, velo_fo + d_velo, velo_bo, &
            state_c, state_b, state_f, state_i, state_o, boundary_flag)) / step
          jac_bo(:, i) = aimag(make_ode(vertex_bi, vertex_fi, vertex_fo, vertex_bo, &
            velo_bi, velo_fi, velo_fo, velo_bo + d_velo, &
            state_c, state_b, state_f, state_i, state_o, boundary_flag)) / step
        end do
      end associate
    end do
  end subroutine compute_odes_wrt_velocities

  !> @brief Computes forces
  !!
  !! The section forces on the airfoil are `𝓕 = 𝓕(𝓤,𝓧)`, where `𝓤` are the states, and `𝓧` are the
  !! grid vertex coordinates. To get the classical force coefficients, the forces need to be summed
  !! and be divided by `Maₒₒ²⋅γ⋅c/2`, where `c` it the chord expressed in grid units.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] vertices Grid vertex coordinate
  !! @param[in] states States
  !! @param[inout] forces section forces
  pure subroutine compute_forces(num_radial, num_angular, vertices, states, forces)
    !f2py integer, intent(hide), depend(states) :: num_radial = size(states, 2)
    !f2py integer, intent(hide), depend(states) :: num_angular = size(states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2
    integer, intent(in) :: num_radial, num_angular
    complex(8), intent(in) :: vertices(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: states(num_var, num_radial, num_angular)
    complex(8), intent(inout) :: forces(num_dim, num_angular)
    integer :: n
    do concurrent (n = 1:num_angular)
      associate(&
        state => states(:, 1, n), &
        vertex_bi => vertices(:, 1, n), &
        vertex_fi => vertices(:, 1, n + 1)&
        )
        forces(:, n) = get_pressure(state) * make_normal(vertex_bi, vertex_fi)
      end associate
    end do
  end subroutine compute_forces

  !> @brief Computes jacobians of ``compute_forces`` with respect to ``states``
  !!
  !! The jacobians `∂𝓕/∂𝓤` are to be used in
  !! * apply_forces_wrt_states_fwd, and
  !! * apply_forces_wrt_states_rev.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] vertices Grid vertex coordinate
  !! @param[in] velocities Grid vertex velocities
  !! @param[in] states States
  !! @param[inout] jacs Jacobians
  pure subroutine compute_forces_wrt_states(num_radial, num_angular, vertices, states, jacs)
    !f2py integer, intent(hide), depend(states) :: num_radial = size(states, 2)
    !f2py integer, intent(hide), depend(states) :: num_angular = size(states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2
    integer, intent(in) :: num_radial, num_angular
    complex(8), intent(in) :: vertices(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: states(num_var, num_radial, num_angular)
    real(8), intent(inout) :: jacs(num_dim, num_var, num_angular)
    complex(8) :: d_state(num_var)
    integer :: i, n
    do concurrent (n = 1:num_angular)
      associate(&
        state => states(:, 1, n), &
        vertex_bi => vertices(:, 1, n), &
        vertex_fi => vertices(:, 1, n + 1), &
        jac => jacs(:, :, n)&
        )
        do concurrent (i = 1:num_var)
          d_state = 0
          d_state(i) = i_step
          jac(:, i) = aimag(get_pressure(state + d_state) * make_normal(vertex_bi, vertex_fi)) &
            / step
        end do
      end associate
    end do
  end subroutine compute_forces_wrt_states

  !> @brief Computes jacobians of ``compute_forces`` with respect to ``vertices``
  !!
  !! The jacobians `∂𝓕/∂𝓧` are to be used in
  !! * apply_forces_wrt_vertices_fwd, and
  !! * apply_forces_wrt_vertices_rev.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] vertices Grid vertex coordinate
  !! @param[in] velocities Grid vertex velocities
  !! @param[in] states States
  !! @param[inout] jacs Jacobians
  pure subroutine compute_forces_wrt_vertices(num_radial, num_angular, vertices, states, jacs)
    !f2py integer, intent(hide), depend(states) :: num_radial = size(states, 2)
    !f2py integer, intent(hide), depend(states) :: num_angular = size(states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_force_wrt_vertex = 2
    integer, intent(in) :: num_radial, num_angular
    complex(8), intent(in) :: vertices(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(in) :: states(num_var, num_radial, num_angular)
    real(8), intent(inout) :: jacs(num_dim, num_dim, num_angular, num_force_wrt_vertex)
    complex(8) :: d_vertex(num_dim)
    integer :: i, n
    jacs = 0
    do concurrent (n = 1:num_angular)
      associate(&
        state => states(:, 1, n), &
        vertex_bi => vertices(:, 1, n), &
        vertex_fi => vertices(:, 1, n + 1), &
        jac_bi => jacs(:, :, n, 1), &
        jac_fi => jacs(:, :, n, 2)&
        )
        do concurrent (i = 1:num_dim)
          d_vertex = 0
          d_vertex(i) = i_step
          jac_bi(:, i) = &
            aimag(get_pressure(state) * make_normal(vertex_bi + d_vertex, vertex_fi)) / step
          jac_fi(:, i) = &
            aimag(get_pressure(state) * make_normal(vertex_bi, vertex_fi + d_vertex)) / step
        end do
      end associate
    end do
  end subroutine compute_forces_wrt_vertices

  !> @brief Applies jacobians of ``compute_forces`` with respect to ``vertices`` in forward mode
  !!
  !! Computes the matrix-vector-product `∂𝓕/∂𝓧⋅δ𝓧`, i.e. the directional derivative.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] jacs Jacobians from ``compute_forces_wrt_vertices``
  !! @param[in] d_vertices Vector to multiply to the Jacobians
  !! @param[inout] d_forces Vector-product
  pure subroutine apply_forces_wrt_vertices_fwd(&
    num_radial, num_angular, jacs, d_vertices, d_forces)
    !f2py integer, intent(hide) :: num_radial
    !f2py integer, intent(hide) :: num_angular
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_force_wrt_vertex = 2
    integer, intent(in) :: num_radial, num_angular
    real(8), intent(in) :: jacs(num_dim, num_dim, num_angular, num_force_wrt_vertex)
    complex(8), intent(in) :: d_vertices(num_dim, num_radial + 1, num_angular + 1)
    complex(8), intent(inout) :: d_forces(num_dim, num_angular)
    integer :: n
    do concurrent (n = 1:num_angular)
      associate (&
        d_vertex_bi => d_vertices(:, 1, n), &
        d_vertex_fi => d_vertices(:, 1, n + 1), &
        jac_bi => jacs(:, :, n, 1), &
        jac_fi => jacs(:, :, n, 2), &
        d_force => d_forces(:, n)&
        )
        d_force = matmul(jac_bi, d_vertex_bi) + matmul(jac_fi, d_vertex_fi)
      end associate
    end do
  end subroutine apply_forces_wrt_vertices_fwd

  !> @brief Applies jacobians of ``compute_forces`` with respect to ``vertices`` in reverse mode
  !!
  !! Computes the matrix-vector-product `∂𝓕/∂𝓧ᵀ⋅δ𝓕`.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] jacs Jacobians from ``compute_forces_wrt_vertices``
  !! @param[in] d_vertices Covector to multiply to the Jacobians
  !! @param[inout] d_forces Covector-product
  pure subroutine apply_forces_wrt_vertices_rev(&
    num_radial, num_angular, jacs, d_forces, d_vertices)
    !f2py integer, intent(hide) :: num_radial
    !f2py integer, intent(hide) :: num_angular
    !f2py integer, parameter :: num_var = 4, num_dim = 2, num_force_wrt_vertex = 2
    integer, intent(in) :: num_radial, num_angular
    real(8), intent(in) :: jacs(num_dim, num_dim, num_angular, num_force_wrt_vertex)
    complex(8), intent(in) :: d_forces(num_dim, num_angular)
    complex(8), intent(inout) :: d_vertices(num_dim, num_radial + 1, num_angular + 1)
    integer :: n
    d_vertices = 0
    do concurrent (n = 1:num_angular)
      associate (&
        d_force => d_forces(:, n), &
        jac_bi => jacs(:, :, n, 1), &
        jac_fi => jacs(:, :, n, 2), &
        d_vertex_bi => d_vertices(:, 1, n), &
        d_vertex_fi => d_vertices(:, 1, n + 1)&
        )
        d_vertex_bi = d_vertex_bi + matmul(d_force, jac_bi)
        d_vertex_fi = d_vertex_fi + matmul(d_force, jac_fi)
      end associate
    end do
  end subroutine apply_forces_wrt_vertices_rev

  !> @brief Applies jacobians of ``compute_forces`` with respect to ``states`` in forward mode
  !!
  !! Computes the matrix-vector-product `∂𝓕/∂𝓤⋅δ𝓤`, i.e. the directional derivative.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] jacs Jacobians from ``compute_forces_wrt_vertices``
  !! @param[in] d_states Vector to multiply to the Jacobians
  !! @param[inout] d_forces Vector-product
  pure subroutine apply_forces_wrt_states_fwd(&
    num_radial, num_angular, jacs, d_states, d_forces)
    !f2py integer, intent(hide), depend(d_states) :: num_radial = size(d_states, 2)
    !f2py integer, intent(hide), depend(d_states) :: num_angular = size(d_states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2
    integer, intent(in) :: num_radial, num_angular
    real(8), intent(in) :: jacs(num_dim, num_var, num_angular)
    complex(8), intent(in) :: d_states(num_var, num_radial, num_angular)
    complex(8), intent(inout) :: d_forces(num_dim, num_angular)
    integer :: n
    do concurrent (n = 1:num_angular)
      associate (d_state => d_states(:, 1, n), jac => jacs(:, :, n), d_force => d_forces(:, n))
        d_forces(:, n) = matmul(jac, d_state)
      end associate
    end do
  end subroutine apply_forces_wrt_states_fwd

  !> @brief Applies jacobians of ``compute_forces`` with respect to ``states`` in reverse mode
  !!
  !! Computes the matrix-vector-product `∂𝓕/∂𝓤ᵀ⋅δ𝓕`.
  !!
  !! @param[in] num_radial Number of cells in the radial direction
  !! @param[in] num_angular Number of cells in the angular direction
  !! @param[in] jacs Jacobians from ``compute_forces_wrt_vertices``
  !! @param[in] d_forces Covector to multiply to the Jacobians
  !! @param[inout] d_states Covector-product
  pure subroutine apply_forces_wrt_states_rev(&
    num_radial, num_angular, jacs, d_forces, d_states)
    !f2py integer, intent(hide), depend(d_states) :: num_radial = size(d_states, 2)
    !f2py integer, intent(hide), depend(d_states) :: num_angular = size(d_states, 3)
    !f2py integer, parameter :: num_var = 4, num_dim = 2
    integer, intent(in) :: num_radial, num_angular
    real(8), intent(in) :: jacs(num_dim, num_var, num_angular)
    complex(8), intent(in) :: d_forces(num_dim, num_angular)
    complex(8), intent(inout) :: d_states(num_var, num_radial, num_angular)
    integer :: n
    d_states = 0
    do concurrent (n = 1:num_angular)
      associate (d_force => d_forces(:, n), jac => jacs(:, :, n), d_state => d_states(:, 1, n))
        d_state = matmul(d_force, jac)
      end associate
    end do
  end subroutine apply_forces_wrt_states_rev

end module spatial_discretization