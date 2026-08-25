module shu_euler_3d_state
  use iso_fortran_env, only: real32, int64
  implicit none

  integer, parameter :: rk = real32
  integer, parameter :: equations = 5, ghosts = 3
  real(rk), parameter :: gamma = 1.4_rk
  real(rk), parameter :: gamma_minus_one = gamma - 1.0_rk
  real(rk), parameter :: weno_epsilon = 1.0e-6_rk
  real(rk), parameter :: lf_enlargement = 1.1_rk

  integer :: nx, ny, nz, maximum_n
  real(rk) :: dx, dy, dz
  real(rk), allocatable :: state(:, :, :, :, :)
  real(rk), allocatable :: rhs(:, :, :, :)

  real(rk), allocatable :: line_state(:, :), line_flux(:, :)
  real(rk), allocatable :: left_eigenvectors(:, :, :)
  real(rk), allocatable :: right_eigenvectors(:, :, :)
  real(rk), allocatable :: flux_difference(:, :), state_difference(:, :)
  real(rk), allocatable :: split_positive(:, :), split_negative(:, :)
  real(rk), allocatable :: characteristic_flux(:, :)
  real(rk), allocatable :: numerical_flux(:, :), line_derivative(:, :)

contains

  subroutine allocate_solver(nx_input, ny_input, nz_input)
    integer, intent(in) :: nx_input, ny_input, nz_input
    integer :: allocation_status
    integer(int64) :: cells, persistent_reals
    character(len=256) :: allocation_message

    if (min(nx_input, ny_input, nz_input) < 4) then
      error stop 'every dimension needs at least four intervals'
    end if
    nx = nx_input
    ny = ny_input
    nz = nz_input
    maximum_n = max(nx, ny, nz)
    dx = 10.0_rk / real(nx, rk)
    dy = 10.0_rk / real(ny, rk)
    dz = 10.0_rk / real(nz, rk)

    allocate( &
      state(-ghosts:nx + ghosts, -ghosts:ny + ghosts, &
            -ghosts:nz + ghosts, equations, 0:3), &
      rhs(0:nx, 0:ny, 0:nz, equations), &
      line_state(equations, -ghosts:maximum_n + ghosts), &
      line_flux(equations, -ghosts:maximum_n + ghosts), &
      left_eigenvectors(-1:maximum_n, equations, equations), &
      right_eigenvectors(-1:maximum_n, equations, equations), &
      flux_difference(equations, -ghosts:maximum_n + 2), &
      state_difference(equations, -ghosts:maximum_n + 2), &
      split_positive(equations, -ghosts:maximum_n + 2), &
      split_negative(equations, -ghosts:maximum_n + 2), &
      characteristic_flux(equations, -1:maximum_n), &
      numerical_flux(equations, -1:maximum_n), &
      line_derivative(equations, 0:maximum_n), &
      stat=allocation_status, errmsg=allocation_message)
    if (allocation_status /= 0) then
      write(*, '(A)') 'allocation failed: ' // trim(allocation_message)
      error stop 2
    end if

    cells = int(nx + 7, int64) * int(ny + 7, int64) * int(nz + 7, int64)
    persistent_reals = 20_int64 * cells + &
      5_int64 * int(nx + 1, int64) * int(ny + 1, int64) * int(nz + 1, int64)
    write(*, '(A,I0,A,I0,A,I0)') 'allocated grid ', nx, ' x ', ny, ' x ', nz
    write(*, '(A,F10.3,A)') 'principal state estimate ', &
      real(4_int64 * persistent_reals, rk) / 1024.0_rk**3, ' GiB'
  end subroutine allocate_solver


  subroutine initialize_extruded_vortex
    integer :: i, j, k
    real(rk) :: x, y, pi, coefficient, exponential, radius_squared
    real(rk) :: temperature, pressure, density, vx, vy, vz, kinetic

    pi = 4.0_rk * atan(1.0_rk)
    coefficient = 5.0_rk / (2.0_rk * pi * exp(-0.5_rk))
    do k = -ghosts, nz + ghosts
      do j = -ghosts, ny + ghosts
        y = real(j, rk) * dy
        do i = -ghosts, nx + ghosts
          x = real(i, rk) * dx
          radius_squared = (x - 5.0_rk)**2 + (y - 5.0_rk)**2
          exponential = exp(-0.5_rk * radius_squared)
          vx = -coefficient * exponential * (y - 5.0_rk)
          vy =  coefficient * exponential * (x - 5.0_rk)
          vz = 0.0_rk
          temperature = 1.0_rk - 0.5_rk * coefficient**2 * &
            exponential**2 * gamma_minus_one / gamma
          pressure = temperature**(gamma / gamma_minus_one)
          density = pressure / temperature
          kinetic = vx**2 + vy**2 + vz**2
          state(i, j, k, 1, 0) = density
          state(i, j, k, 2, 0) = density * vx
          state(i, j, k, 3, 0) = density * vy
          state(i, j, k, 4, 0) = density * vz
          state(i, j, k, 5, 0) = pressure / gamma_minus_one + &
            0.5_rk * density * kinetic
        end do
      end do
    end do
  end subroutine initialize_extruded_vortex


  subroutine apply_periodic_boundary(stage)
    integer, intent(in) :: stage
    integer :: i, j, k, m

    do m = 1, equations
      do k = 0, nz
        do j = 0, ny
          do i = 0, ghosts
            state(-i, j, k, m, stage) = state(nx - i, j, k, m, stage)
            state(nx + i, j, k, m, stage) = state(i, j, k, m, stage)
          end do
        end do
      end do
      do k = 0, nz
        do i = 0, nx
          do j = 0, ghosts
            state(i, -j, k, m, stage) = state(i, ny - j, k, m, stage)
            state(i, ny + j, k, m, stage) = state(i, j, k, m, stage)
          end do
        end do
      end do
      do j = 0, ny
        do i = 0, nx
          do k = 0, ghosts
            state(i, j, -k, m, stage) = state(i, j, nz - k, m, stage)
            state(i, j, nz + k, m, stage) = state(i, j, k, m, stage)
          end do
        end do
      end do
    end do
  end subroutine apply_periodic_boundary


  subroutine prepare_line(number_of_intervals, inverse_spacing)
    integer, intent(in) :: number_of_intervals
    real(rk), intent(in) :: inverse_spacing
    integer :: i, q, field, candidate
    integer :: positive_offset, negative_offset
    real(rk) :: density, normal_momentum, energy, inverse_density
    real(rk) :: velocity(3), pressure, sound_speed, enthalpy
    real(rk) :: alpha(equations), sqrt_density(-ghosts:maximum_n + ghosts)
    real(rk) :: velocity_values(3, -ghosts:maximum_n + ghosts)
    real(rk) :: enthalpy_values(-ghosts:maximum_n + ghosts)
    real(rk) :: fraction, roe_velocity(3), roe_enthalpy, roe_q, roe_sound
    real(rk) :: reciprocal_sound, b1, b2
    real(rk) :: h_positive(4), h_negative(4)

    alpha = 1.0e-15_rk
    do i = -ghosts, number_of_intervals + ghosts
      density = line_state(1, i)
      normal_momentum = line_state(2, i)
      energy = line_state(5, i)
      inverse_density = 1.0_rk / density
      velocity(1) = normal_momentum * inverse_density
      velocity(2) = line_state(3, i) * inverse_density
      velocity(3) = line_state(4, i) * inverse_density
      pressure = gamma_minus_one * (energy - 0.5_rk * density * &
        sum(velocity**2))
      sound_speed = sqrt(gamma * pressure * inverse_density)
      enthalpy = (pressure + energy) * inverse_density

      line_flux(1, i) = normal_momentum
      line_flux(2, i) = velocity(1) * normal_momentum + pressure
      line_flux(3, i) = velocity(1) * line_state(3, i)
      line_flux(4, i) = velocity(1) * line_state(4, i)
      line_flux(5, i) = velocity(1) * (pressure + energy)
      sqrt_density(i) = sqrt(density)
      velocity_values(:, i) = velocity
      enthalpy_values(i) = enthalpy

      alpha(1) = max(alpha(1), abs(velocity(1) - sound_speed))
      alpha(2) = max(alpha(2), abs(velocity(1)))
      alpha(5) = max(alpha(5), abs(velocity(1) + sound_speed))
    end do
    alpha(1) = lf_enlargement * alpha(1)
    alpha(2) = lf_enlargement * alpha(2)
    alpha(3) = alpha(2)
    alpha(4) = alpha(2)
    alpha(5) = lf_enlargement * alpha(5)

    do i = -1, number_of_intervals
      fraction = sqrt_density(i) / (sqrt_density(i) + sqrt_density(i + 1))
      roe_velocity = fraction * velocity_values(:, i) + &
        (1.0_rk - fraction) * velocity_values(:, i + 1)
      roe_enthalpy = fraction * enthalpy_values(i) + &
        (1.0_rk - fraction) * enthalpy_values(i + 1)
      roe_q = 0.5_rk * sum(roe_velocity**2)
      roe_sound = sqrt(gamma_minus_one * (roe_enthalpy - roe_q))

      right_eigenvectors(i, :, 1) = [1.0_rk, &
        roe_velocity(1) - roe_sound, roe_velocity(2), roe_velocity(3), &
        roe_enthalpy - roe_velocity(1) * roe_sound]
      right_eigenvectors(i, :, 2) = [0.0_rk, 0.0_rk, 1.0_rk, 0.0_rk, &
        roe_velocity(2)]
      right_eigenvectors(i, :, 3) = [0.0_rk, 0.0_rk, 0.0_rk, 1.0_rk, &
        roe_velocity(3)]
      right_eigenvectors(i, :, 4) = [1.0_rk, roe_velocity(1), &
        roe_velocity(2), roe_velocity(3), roe_q]
      right_eigenvectors(i, :, 5) = [1.0_rk, &
        roe_velocity(1) + roe_sound, roe_velocity(2), roe_velocity(3), &
        roe_enthalpy + roe_velocity(1) * roe_sound]

      reciprocal_sound = 1.0_rk / roe_sound
      b1 = gamma_minus_one * reciprocal_sound**2
      b2 = roe_q * b1
      left_eigenvectors(i, 1, :) = [ &
        0.5_rk * (b2 + roe_velocity(1) * reciprocal_sound), &
        -0.5_rk * (b1 * roe_velocity(1) + reciprocal_sound), &
        -0.5_rk * b1 * roe_velocity(2), &
        -0.5_rk * b1 * roe_velocity(3), 0.5_rk * b1]
      left_eigenvectors(i, 2, :) = [-roe_velocity(2), 0.0_rk, 1.0_rk, &
        0.0_rk, 0.0_rk]
      left_eigenvectors(i, 3, :) = [-roe_velocity(3), 0.0_rk, 0.0_rk, &
        1.0_rk, 0.0_rk]
      left_eigenvectors(i, 4, :) = [1.0_rk - b2, &
        b1 * roe_velocity(1), b1 * roe_velocity(2), &
        b1 * roe_velocity(3), -b1]
      left_eigenvectors(i, 5, :) = [ &
        0.5_rk * (b2 - roe_velocity(1) * reciprocal_sound), &
        -0.5_rk * (b1 * roe_velocity(1) - reciprocal_sound), &
        -0.5_rk * b1 * roe_velocity(2), &
        -0.5_rk * b1 * roe_velocity(3), 0.5_rk * b1]
    end do

    do q = 1, equations
      do i = -ghosts, number_of_intervals + 2
        flux_difference(q, i) = line_flux(q, i + 1) - line_flux(q, i)
        state_difference(q, i) = line_state(q, i + 1) - line_state(q, i)
      end do
    end do

    do field = 1, equations
      do q = 1, equations
        do i = -ghosts, number_of_intervals + 2
          split_positive(q, i) = 0.5_rk * (flux_difference(q, i) + &
            alpha(field) * state_difference(q, i))
          split_negative(q, i) = split_positive(q, i) - &
            flux_difference(q, i)
        end do
      end do

      do i = -1, number_of_intervals
        do candidate = 1, 4
          positive_offset = candidate - 3
          negative_offset = 3 - candidate
          h_positive(candidate) = sum(left_eigenvectors(i, field, :) * &
            split_positive(:, i + positive_offset))
          h_negative(candidate) = sum(left_eigenvectors(i, field, :) * &
            split_negative(:, i + negative_offset))
        end do
        characteristic_flux(field, i) = nonlinear_correction(h_positive) + &
          nonlinear_correction(h_negative)
      end do
    end do

    do q = 1, equations
      do i = -1, number_of_intervals
        numerical_flux(q, i) = &
          sum(right_eigenvectors(i, q, :) * characteristic_flux(:, i)) + &
          (-line_flux(q, i - 1) + 7.0_rk * &
          (line_flux(q, i) + line_flux(q, i + 1)) - &
          line_flux(q, i + 2)) / 12.0_rk
      end do
      do i = 0, number_of_intervals
        line_derivative(q, i) = &
          (numerical_flux(q, i - 1) - numerical_flux(q, i)) * inverse_spacing
      end do
    end do
  end subroutine prepare_line


  pure real(rk) function nonlinear_correction(h) result(correction)
    real(rk), intent(in) :: h(4)
    real(rk) :: t1, t2, t3, indicator1, indicator2, indicator3
    real(rk) :: denominator1, denominator2, denominator3
    real(rk) :: weight1, weight2, weight3, reciprocal_sum

    t1 = h(1) - h(2)
    t2 = h(2) - h(3)
    t3 = h(3) - h(4)
    indicator1 = 13.0_rk * t1**2 + 3.0_rk * (h(1) - 3.0_rk * h(2))**2
    indicator2 = 13.0_rk * t2**2 + 3.0_rk * (h(2) + h(3))**2
    indicator3 = 13.0_rk * t3**2 + 3.0_rk * (3.0_rk * h(3) - h(4))**2
    denominator1 = (weno_epsilon + indicator1)**2
    denominator2 = (weno_epsilon + indicator2)**2
    denominator3 = (weno_epsilon + indicator3)**2
    weight1 = denominator2 * denominator3
    weight2 = 6.0_rk * denominator1 * denominator3
    weight3 = 3.0_rk * denominator1 * denominator2
    reciprocal_sum = 1.0_rk / (weight1 + weight2 + weight3)
    weight1 = weight1 * reciprocal_sum
    weight3 = weight3 * reciprocal_sum
    correction = (weight1 * (t2 - t1) + &
      (0.5_rk * weight3 - 0.25_rk) * (t3 - t2)) / 3.0_rk
  end function nonlinear_correction


  subroutine compute_rhs(stage)
    integer, intent(in) :: stage
    integer :: i, j, k

    call apply_periodic_boundary(stage)

    do k = 0, nz
      do j = 0, ny
        do i = -ghosts, nx + ghosts
          line_state(:, i) = state(i, j, k, :, stage)
        end do
        call prepare_line(nx, 1.0_rk / dx)
        do i = 0, nx
          rhs(i, j, k, :) = line_derivative(:, i)
        end do
      end do
    end do

    do k = 0, nz
      do i = 0, nx
        do j = -ghosts, ny + ghosts
          line_state(:, j) = state(i, j, k, [1, 3, 2, 4, 5], stage)
        end do
        call prepare_line(ny, 1.0_rk / dy)
        do j = 0, ny
          rhs(i, j, k, 1) = rhs(i, j, k, 1) + line_derivative(1, j)
          rhs(i, j, k, 2) = rhs(i, j, k, 2) + line_derivative(3, j)
          rhs(i, j, k, 3) = rhs(i, j, k, 3) + line_derivative(2, j)
          rhs(i, j, k, 4) = rhs(i, j, k, 4) + line_derivative(4, j)
          rhs(i, j, k, 5) = rhs(i, j, k, 5) + line_derivative(5, j)
        end do
      end do
    end do

    do j = 0, ny
      do i = 0, nx
        do k = -ghosts, nz + ghosts
          line_state(:, k) = state(i, j, k, [1, 4, 2, 3, 5], stage)
        end do
        call prepare_line(nz, 1.0_rk / dz)
        do k = 0, nz
          rhs(i, j, k, 1) = rhs(i, j, k, 1) + line_derivative(1, k)
          rhs(i, j, k, 2) = rhs(i, j, k, 2) + line_derivative(3, k)
          rhs(i, j, k, 3) = rhs(i, j, k, 3) + line_derivative(4, k)
          rhs(i, j, k, 4) = rhs(i, j, k, 4) + line_derivative(2, k)
          rhs(i, j, k, 5) = rhs(i, j, k, 5) + line_derivative(5, k)
        end do
      end do
    end do
  end subroutine compute_rhs


  real(rk) function compute_timestep(cfl) result(dt)
    real(rk), intent(in) :: cfl
    integer :: i, j, k
    real(rk) :: density, velocity(3), pressure, sound_speed, local_speed
    real(rk) :: maximum_speed

    maximum_speed = 0.0_rk
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          density = state(i, j, k, 1, 0)
          velocity = state(i, j, k, 2:4, 0) / density
          pressure = gamma_minus_one * (state(i, j, k, 5, 0) - &
            0.5_rk * density * sum(velocity**2))
          sound_speed = sqrt(gamma * pressure / density)
          local_speed = (abs(velocity(1)) + sound_speed) / dx + &
            (abs(velocity(2)) + sound_speed) / dy + &
            (abs(velocity(3)) + sound_speed) / dz
          maximum_speed = max(maximum_speed, local_speed)
        end do
      end do
    end do
    dt = cfl / maximum_speed
  end function compute_timestep


  subroutine rk3_step(dt)
    real(rk), intent(in) :: dt
    integer :: i, j, k, m

    call compute_rhs(0)
    do m = 1, equations
      do k = 0, nz
        do j = 0, ny
          do i = 0, nx
            state(i, j, k, m, 1) = state(i, j, k, m, 0) + &
              dt * rhs(i, j, k, m)
          end do
        end do
      end do
    end do

    call compute_rhs(1)
    do m = 1, equations
      do k = 0, nz
        do j = 0, ny
          do i = 0, nx
            state(i, j, k, m, 2) = 0.75_rk * state(i, j, k, m, 0) + &
              0.25_rk * (state(i, j, k, m, 1) + dt * rhs(i, j, k, m))
          end do
        end do
      end do
    end do

    call compute_rhs(2)
    do m = 1, equations
      do k = 0, nz
        do j = 0, ny
          do i = 0, nx
            state(i, j, k, m, 0) = (state(i, j, k, m, 0) + &
              2.0_rk * (state(i, j, k, m, 2) + dt * rhs(i, j, k, m))) / &
              3.0_rk
          end do
        end do
      end do
    end do
    call apply_periodic_boundary(0)
  end subroutine rk3_step


  subroutine write_state_if_requested(time)
    real(rk), intent(in) :: time
    character(len=512) :: output_path
    integer :: environment_status, unit_number, i, j, k, m

    call get_environment_variable('WENO_WRITE_STATE', output_path, &
      status=environment_status)
    if (environment_status /= 0 .or. len_trim(output_path) == 0) return
    open(newunit=unit_number, file=trim(output_path), access='stream', &
      form='unformatted', status='replace')
    write(unit_number) time, &
      ((((state(i, j, k, m, 0), i=0,nx), j=0,ny), k=0,nz), &
      m=1,equations)
    close(unit_number)
  end subroutine write_state_if_requested

end module shu_euler_3d_state


program shu_euler_3d
  use shu_euler_3d_state
  implicit none
  integer :: steps, step, completed_steps
  real(rk) :: cfl, final_time, time, dt

  read(*, *) nx, ny, nz
  read(*, *) cfl
  read(*, *) steps
  read(*, *) final_time
  call allocate_solver(nx, ny, nz)
  call initialize_extruded_vortex
  time = 0.0_rk
  completed_steps = 0

  do step = 1, steps
    dt = compute_timestep(cfl)
    if (time + dt >= final_time) dt = final_time - time
    time = time + dt
    call rk3_step(dt)
    completed_steps = step
    if (time >= final_time) exit
  end do
  call write_state_if_requested(time)
  write(*, '(A,I0,A,ES14.7)') 'steps = ', completed_steps, ' time = ', time
end program shu_euler_3d
