module weno_state
  use iso_fortran_env, only: int64
  implicit none

  integer, parameter :: md = 3, mnm = 4

  integer :: ntot, mt, mn, nx, nxm, ny, nym, nt
  real :: gamma, gm1, cfl, epweno, tend
  real :: dx, cdx, dy, cdy, dt, em, tnum, velocity
  real :: tcpu = 0.0, t00 = 0.0

  real, allocatable :: x(:), y(:)
  real, allocatable :: uc(:, :, :, :), rhs(:, :, :)
  real, allocatable :: u(:, :), f(:, :)
  real, allocatable :: evl(:, :, :), evr(:, :, :)
  real, allocatable :: am(:), fh(:, :)

  logical :: write_solution = .true.

contains

  subroutine allocate_state(nx_points, ny_points)
    integer, intent(in) :: nx_points, ny_points
    integer :: allocation_status, environment_status, max_points
    integer(int64) :: x_count, y_count, line_count, persistent_reals
    character(len=256) :: allocation_message
    character(len=32) :: environment_value

    if (nx_points < 1 .or. ny_points < 1) then
      error stop 'nx and ny must both be positive'
    end if

    nxm = nx_points + md
    nym = ny_points + md
    max_points = max(nx_points, ny_points)

    allocate( &
      x(-md:nxm), y(-md:nym), &
      uc(-md:nxm, -md:nym, mnm, 0:3), &
      rhs(-md:nxm, -md:nym, mnm), &
      u(-md:max_points + md, mnm), &
      f(-md:max_points + md, mnm), &
      evl(-md:max_points + md, mnm, mnm), &
      evr(-md:max_points + md, mnm, mnm), &
      am(mnm), fh(-md:max_points + md, mnm), &
      stat=allocation_status, errmsg=allocation_message)

    if (allocation_status /= 0) then
      write(*, '(A)') 'state allocation failed: ' // trim(allocation_message)
      error stop 2
    end if

    x_count = int(nx_points + 2 * md + 1, int64)
    y_count = int(ny_points + 2 * md + 1, int64)
    line_count = int(max_points + 2 * md + 1, int64)
    persistent_reals = x_count + y_count + &
      20_int64 * x_count * y_count + 44_int64 * line_count + 4_int64

    write(*, '(A,1X,I0,A,I0)') 'allocated grid', nx_points, ' x ', ny_points
    write(*, '(A,1X,F12.3,A)') 'persistent state estimate', &
      real(4_int64 * persistent_reals) / 1024.0**3, ' GiB'

    call get_environment_variable( &
      'WENO_WRITE_SOLUTION', environment_value, status=environment_status)
    if (environment_status == 0 .and. trim(environment_value) == '0') then
      write_solution = .false.
    end if

    call get_environment_variable( &
      'WENO_TOUCH_ALL', environment_value, status=environment_status)
    if (environment_status == 0 .and. trim(environment_value) == '1') then
      x = 0.0
      y = 0.0
      uc = 0.0
      rhs = 0.0
      u = 0.0
      f = 0.0
      evl = 0.0
      evr = 0.0
      am = 0.0
      fh = 0.0
    end if
  end subroutine allocate_state

end module weno_state
