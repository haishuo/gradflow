c
c program: weno.f
c
c purpose: to solve the two dimensional Euler
c          equations of compressible gas dynamics in a rectangle
c          using WENO method.
c
c reference: Efficient implementation of weighted ENO schemes,
c            by Guang-Shan Jiang and Chi-Wang Shu,
c            in Journal of Computational Physics, v126 (1996), pp.202-228.
c
c The code is set up to run a periodic vortex problem.
c However, it can be easily changed to run other problems
c
c this code runs optimally on Cray.  The second()
c and the ismax functions are special for CRAY (should
c comment and uncomment certain lines around these functions before
c running on CRAY).
c
c the result is in the file fort.8,  in the
c order of   x, y, density, pressure, x-velocity, y-velocity
c and in the file fort.9 for the 1D cut at x=xhalf,  in the order of
c y, density, pressure, x-velocity, y-velocity

      program WENO_LF
      include 'comm.inc'
*****678****************************************************************
* Name:      main.f
* Function:  drive routine
*            System to solve: u_t + f(u)_x + g(u)_y = 0
*   or
*            u_t = RHS = -f(u)_x - g(u)_y
*****678****************************************************************

* readin parameters

      call setup

* initialization

      call init

* 'io' is the main controller which rotates between 0 and "mt-1"
* corresponding to the different stage in Runge-Kutta schemes

      io = 0
      istop = 0
      nt = 0

      if(ntot.eq.0) goto 1001

*****678*************** begin time evolution ***************************

1000  continue

* impose boundary condition

call bc(io)

      if( io.eq.0 .and. istop.eq.1) goto 1001
c       if(io.eq.0 .and. second().gt.550.) goto 1001

* compute -f(u)_x

call fx(io)

* compute -g(u)_y

call gy(io)

* compute time step size "dt"

      if( io.eq.0 ) then

c      tcpu=second()
      if(nt/20*20.eq.nt) then
      write(6,*) nt,'t=',tnum,'accumulated cpu=',tcpu
      write(6,*) 'cpu per 20 time steps',tcpu-t00
      t00=tcpu
      end if
call cflc(aam)
      dt=cfl/aam

c         dt = cfl / em / ( cdx + cdy )
      if( ( tnum + dt ) .ge. tend ) then
      dt = tend - tnum
      istop = 1
      endif
      tnum = tnum + dt
      nt = nt + 1
      if( nt.ge.ntot ) istop = 1
      endif

* Runge-Kutta scheme for advancing in time

call rk(io)

      io = mod( io+1, mt )

      goto 1000

*****678***************  end  time evolution ***************************

1001  continue

      write(*,*) 'nt = ', nt, '  tnum = ', tnum

* output

      call save

      stop
      end


      subroutine setup
      include 'comm.inc'
*****678****************************************************************
* Name:      setup.f
* Function:  set up the # of grid points and read in some parameters
*****678****************************************************************

      write(*,*) ' order in time ? (3,4) '
      read(*,*) mt

c       write(*,*) ' # of equations ? (1,2,3,4) '
c       read(*,*) mn
      mn=4

      write(*,*) ' # of points in x-dir & y-dir ? '
      read(*,*) nx,ny
      nxm = nx + md
      nym = ny + md

      write(*,*) ' cfl ? ( e.g. 0.6 )'
      read(*,*) cfl

      write(*,*) ' maximum # of time steps ? '
      read(*,*) ntot

      write(*,*) ' '
      write(*,*) ' WENO-LF-4 or WENO-LF-5 with RK-3/RK-4 '
      write(*,*) ' '
      write(*,*) 'order in time: ', mt
      write(*,*) '# of equations: ', mn
      write(*,*) '# of points in x-dir: ', nx
      write(*,*) '# of points in y-dir: ', ny
      write(*,*) 'CFL = ', cfl
      write(*,*) 'maximum # of time steps ', ntot

      gamma = 1.4
      gm1 = gamma - 1.0

      epweno = 1.e-6

      return
      end


      subroutine init
      include 'comm.inc'
*****678****************************************************************
* Name:      init_per.f
* Function:  set up the grid and initial condition ( u(x,y) at t= 0 )
*****678****************************************************************

      tend = 0.5
      write(6,*) 'the final time'
      read(5,*) tend

      write(6,*) 'enter problem number'
      write(6,*) '  1 = steady vortex'
      write(6,*) '  2 = horizontally moving vortex'
      write(6,*) '  3 = diagonally moving vortex'
      read(5,*) nprob

      xleft = 0.
      xright = 10.0
      yleft = 0.
      yright = 10.0

      dx = ( xright - xleft ) / nx
cdx = 1./dx
      dy = ( yright - yleft ) / ny
cdy = 1./dy

      do i = -md, nxm
      x(i) = xleft + i * dx
      enddo

      do j = -md, nym
      y(j) = yleft + j * dy
      enddo

      pi = atan(1.0) * 4.0

c rkap is the strength of the vortex

      rkap=5.
coefk=rkap/(2.*pi*exp(-0.5))
      gga=1./(gamma-1.)
      ggb=gamma/(gamma-1.)
      ggc=1.0/ggb
      ggd=-1.0/gamma

      if(nprob.eq.1) then

      dri=1.
      uri=0.
      vri=0.
      pri=1.

      else if(nprob.eq.2) then

      dri=1.
      uri=1.
      vri=0.
      pri=1.

      else if(nprob.eq.3) then

      dri=1.
      uri=1.
      vri=1.
      pri=1.

      end if

      do 1 j=-md,nym
      do 1 i=-md,nxm
cij=dri
      rij=uri
      sij=vri
      pij=pri

      xr=x(i)-5.
      yr=y(j)-5.
      rr2=xr**2+yr**2
      rr=sqrt(rr2)

      eee=exp(-0.5*rr2)
      pold=pij
      told=pij/cij
      entr=pij/cij**gamma
      rij=rij-coefk*eee*yr
      sij=sij+coefk*eee*xr
      tper=-0.5*coefk**2*eee*eee*ggc
      trin=told+tper

      pij=trin**ggb/entr**gga
      cij=pij/trin

      uc(i,j,1,0)=cij
      uc(i,j,2,0)=cij*rij
      uc(i,j,3,0)=cij*sij
      uc(i,j,4,0)=pij/(gamma-1.)+0.5*cij*(rij**2+sij**2)

1     continue

      write(6,*) 'enter the status of the initial condition'
      write(6,*) '  0 = initial at t=0.'
      write(6,*) '  1 = restart from a restart file'
      read(5,*) irest

      tnum = 0.0

      if(irest.eq.1) then
      open(unit=10,file='restart',form='unformatted',status='unknown')
      rewind 10
      read(10) tnum,(((uc(i,j,m,0),i=0,nx),j=0,ny),m=1,4)
      close(10)
      end if

      return
      end


      subroutine bc(io)
      include 'comm.inc'
*****678****************************************************************
* Name:      bc_per.f
* Function:  periodic boundary condition
*****678****************************************************************

      do m = 1, mn

      do i = 0, md
      do j = 0, ny
      uc(  -i,j,m,io) = uc(nx-i,j,m,io)
      uc(nx+i,j,m,io) = uc(   i,j,m,io)
      enddo
      enddo

      do j = 0, md
      do i = 0, nx
      uc(i,  -j,m,io) = uc(i,ny-j,m,io)
      uc(i,ny+j,m,io) = uc(i,   j,m,io)
      enddo
      enddo

      enddo

      return
      end


      subroutine fx(io)
      include 'comm.inc'
*****678****************************************************************
* Name:      fx.f
* Function:  approximate "-df/dx"
*****678****************************************************************

      dimension w(-md:nxd), vx(-md:nxd), vy(-md:nxd), h(-md:nxd)

      em = 1.e-15

*--------------------- begin of outer loop in y-dir. -------------------

      do j = 0, ny

* compute the f(u) in x-direction

      am(1) = 1.e-15
      am(2) = 1.e-15
      am(4) = 1.e-15
      do  i = -md, nxm
      den = uc(i,j,1,io)
      xmt = uc(i,j,2,io)
      ymt = uc(i,j,3,io)
      eng = uc(i,j,4,io)
      t0 = 1. / den
      vex = xmt * t0
      vey = ymt * t0
      pre = gm1 * ( eng - 0.5 * ( xmt * vex + ymt * vey ) )
cvel = sqrt( gamma * pre * t0 )
      u(i,1) = den
      u(i,2) = xmt
      u(i,3) = ymt
      u(i,4) = eng
      f(i,1) = xmt
      f(i,2) = vex * xmt + pre
      f(i,3) = vex * ymt
      f(i,4) = vex * ( pre + eng )
      w(i) = sqrt( den )
      h(i) = ( pre + eng ) * t0
      vx(i) = vex
      vy(i) = vey
      am(1) = max( am(1), abs( vex - cvel ) )
      am(2) = max( am(2), abs( vex        ) )
      am(4) = max( am(4), abs( vex + cvel ) )
      enddo
      am(1) = am(1) * 1.1
      am(2) = am(2) * 1.1
      am(3) = am(2)
      am(4) = am(4) * 1.1
      em = max( em, max( am(1), am(4) ) )

* compute left and right eigenvectors of Roe's mean matrix

      do i = -1, nx
      t0 = w(i) / ( w(i) + w(i+1) )
      t1 = 1. - t0
      vxm = t0 * vx(i) + t1 * vx(i+1)
      vym = t0 * vy(i) + t1 * vy(i+1)
      hm = t0 *  h(i) + t1 *  h(i+1)
      qm = 0.5 * ( vxm * vxm + vym * vym )
cm = sqrt( gm1 * ( hm - qm ) )
      t0 = vxm * cm
      evr(i,1,1) = 1.0
      evr(i,1,2) = 0.0
      evr(i,1,3) = 1.0
      evr(i,1,4) = 1.0
      evr(i,2,1) = vxm - cm
      evr(i,2,2) = 0.0
      evr(i,2,3) = vxm
      evr(i,2,4) = vxm + cm
      evr(i,3,1) = vym
      evr(i,3,2) = 1.0
      evr(i,3,3) = vym
      evr(i,3,4) = vym
      evr(i,4,1) = hm - t0
      evr(i,4,2) = vym
      evr(i,4,3) = qm
      evr(i,4,4) = hm + t0
      rcm = 1. / cm
      b1 = gm1 * rcm * rcm
      b2 = qm * b1
      t0 = vxm * rcm
      t1 = b1 * vxm
      t2 = 0.5 * b1
      t3 = b1 * vym
      evl(i,1,1) = 0.5 * ( b2 + t0 )
      evl(i,1,2) = -0.5 * ( t1 + rcm )
      evl(i,1,3) = -0.5 * t3
      evl(i,1,4) = t2
      evl(i,2,1) = - vym
      evl(i,2,2) = 0.0
      evl(i,2,3) = 1.0
      evl(i,2,4) = 0.0
      evl(i,3,1) = 1. - b2
      evl(i,3,2) = t1
      evl(i,3,3) = t3
      evl(i,3,4) = -b1
      evl(i,4,1) =  0.5 * ( b2 - t0 )
      evl(i,4,2) = -0.5 * ( t1 - rcm )
      evl(i,4,3) = -0.5 *   t3
      evl(i,4,4) = t2
      enddo

* call the WENO slover

call weno(nx)

      do m = 1, mn
      do i = 0, nx
      rhs(i,j,m) = ( fh(i-1,m) - fh(i,m) ) * cdx
      enddo
      enddo

      enddo
*---------------------  end  of outer loop in y-dir. -------------------

      return
      end


      subroutine gy(io)
      include 'comm.inc'
*****678****************************************************************
* Name:      gy.f
* Function:  approximate "-dg/dy"
*****678****************************************************************

      dimension w(-md:nyd), vx(-md:nyd), vy(-md:nyd), h(-md:nyd)

*--------------------- begin of outer loop in x-dir. -------------------

      do j = 0, nx

* compute g(u) in y-direction

      am(1) = 1.e-15
      am(2) = 1.e-15
      am(4) = 1.e-15
      do  i = -md, nym
      den = uc(j,i,1,io)
      xmt = uc(j,i,2,io)
      ymt = uc(j,i,3,io)
      eng = uc(j,i,4,io)
      t0 = 1. / den
      vex = xmt * t0
      vey = ymt * t0
      pre = gm1 * ( eng - 0.5 * ( xmt * vex + ymt * vey ) )
cvel = sqrt( gamma * pre * t0 )
      u(i,1) = den
      u(i,2) = xmt
      u(i,3) = ymt
      u(i,4) = eng
      f(i,1) = ymt
      f(i,2) = vey * xmt
      f(i,3) = vey * ymt + pre
      f(i,4) = vey * ( pre + eng )
      w(i) = sqrt( den )
      h(i) = ( pre + eng ) * t0
      vx(i) = vex
      vy(i) = vey
      am(1) = max( am(1), abs( vey - cvel ) )
      am(2) = max( am(2), abs( vey        ) )
      am(4) = max( am(4), abs( vey + cvel ) )
      enddo
      am(1) = am(1) * 1.1
      am(2) = am(2) * 1.1
      am(3) = am(2)
      am(4) = am(4) * 1.1
      em = max( em, max( am(1), am(4) ) )

* compute left and right eigenvectors of Roe's mean matrix

      do i = -1, ny
      t0 = w(i) / ( w(i) + w(i+1) )
      t1 = 1. - t0
      vxm = t0 * vx(i) + t1 * vx(i+1)
      vym = t0 * vy(i) + t1 * vy(i+1)
      hm = t0 *  h(i) + t1 *  h(i+1)
      qm = 0.5 * ( vxm * vxm + vym * vym )
cm = sqrt( gm1 * ( hm - qm ) )
      t0 = vym * cm
      evr(i,1,1) = 1.0
      evr(i,1,2) = 0.0
      evr(i,1,3) = 1.0
      evr(i,1,4) = 1.0
      evr(i,2,1) = vxm
      evr(i,2,2) = 1.0
      evr(i,2,3) = vxm
      evr(i,2,4) = vxm
      evr(i,3,1) = vym - cm
      evr(i,3,2) = 0.0
      evr(i,3,3) = vym
      evr(i,3,4) = vym + cm
      evr(i,4,1) = hm - t0
      evr(i,4,2) = vxm
      evr(i,4,3) = qm
      evr(i,4,4) = hm + t0
      rcm = 1. / cm
      b1 = gm1 * rcm * rcm
      b2 = qm * b1
      t0 = vym * rcm
      t1 = b1 * vym
      t2 = 0.5 * b1
      t3 = b1 * vxm
      evl(i,1,1) = 0.5 * ( b2 + t0 )
      evl(i,1,2) = -0.5 * t3
      evl(i,1,3) = -0.5 * ( t1 + rcm )
      evl(i,1,4) = t2
      evl(i,2,1) = - vxm
      evl(i,2,2) = 1.0
      evl(i,2,3) = 0.0
      evl(i,2,4) = 0.0
      evl(i,3,1) = 1. - b2
      evl(i,3,2) = t3
      evl(i,3,3) = t1
      evl(i,3,4) = -b1
      evl(i,4,1) =  0.5 * ( b2 - t0 )
      evl(i,4,2) = -0.5 *   t3
      evl(i,4,3) = -0.5 * ( t1 - rcm )
      evl(i,4,4) = t2
      enddo

* call the WENO solver

call weno(ny)

      do m = 1, mn
      do i = 0, ny
      rhs(j,i,m) = rhs(j,i,m) + ( fh(i-1,m) - fh(i,m) ) * cdy
      enddo
      enddo

      enddo
*---------------------  end  of outer loop in x-dir. -------------------

      return
      end


      subroutine weno(ns)
      include 'comm.inc'
*****678****************************************************************
* Name:      wenolf.f
* Function:  Use WENO-LF-4 or WENO-LF-5 to approximate fluxes
*****678****************************************************************

      dimension df(-md:nsd,mnm), du(-md:nsd,mnm)
      dimension ff(-md:nsd,mnm), gg(-md:nsd,mnm,2), hh(-md:nsd,4,2)

      nsm = ns + md

      do m = 1, mn
      do i = -md, nsm - 1
      df(i,m) = f(i+1,m) - f(i,m)
      du(i,m) = u(i+1,m) - u(i,m)
      enddo
      enddo

*----------------- loop in "m" starts here  -------------------

      do m = 1, mn

* use Lax-Friedrichs building block to split the fluxes

      do m1 =  1, mn
      do  i =-md, nsm - 1
      gg(i,m1,1) = 0.5 * ( df(i,m1) + am(m) * du(i,m1) )
      gg(i,m1,2) = gg(i,m1,1) - df(i,m1)
      enddo
      enddo

* Project the positive and negative part of the fluxes to the
* 'm'th characteristic field

      do m1 = 1, 4
      k0 = m1 - 3
      k1 =  3 - m1
      do  i = -1, ns
      hh(i,m1,1) = evl(i,m,1)*gg(i+k0,1,1) + evl(i,m,2)*gg(i+k0,2,1)
     *+ evl(i,m,3)*gg(i+k0,3,1) + evl(i,m,4)*gg(i+k0,4,1)
      hh(i,m1,2) = evl(i,m,1)*gg(i+k1,1,2) + evl(i,m,2)*gg(i+k1,2,2)
     *+ evl(i,m,3)*gg(i+k1,3,2) + evl(i,m,4)*gg(i+k1,4,2)
      enddo
      enddo

* compute the weights and approximate the fluxes

      do i = -1, ns
      ff(i,m) = 0.0
      enddo

      do m1 = 1, 2
      do i = -1, ns

      t1 = hh(i,1,m1) - hh(i,2,m1)
      t2 = hh(i,2,m1) - hh(i,3,m1)
      t3 = hh(i,3,m1) - hh(i,4,m1)

C un-comment(comment) the following 3 lines
C to use (not to use) Liu's smoothness measurement:

*     tt1 = t1**2 + 0.5 * ( hh(i,1,m1)**2 + hh(i,2,m1)**2 )
*     tt2 = t2**2 + 0.5 * ( hh(i,2,m1)**2 + hh(i,3,m1)**2 )
*     tt3 = t3**2 + 0.5 * ( hh(i,3,m1)**2 + hh(i,4,m1)**2 )

C un-comment(comment) the following 3 lines
C to use (not to use) the new smoothness measurement:

      tt1 = 13. * t1**2 + 3. * (   hh(i,1,m1) - 3*hh(i,2,m1) )**2
      tt2 = 13. * t2**2 + 3. * (   hh(i,2,m1) +   hh(i,3,m1) )**2
      tt3 = 13. * t3**2 + 3. * ( 3*hh(i,3,m1) -   hh(i,4,m1) )**2

      tt1 =  ( epweno + tt1 )**2
      tt2 =  ( epweno + tt2 )**2
      tt3 =  ( epweno + tt3 )**2
      s1 =      tt2 * tt3
      s2 = 6. * tt1 * tt3
      s3 = 3. * tt1 * tt2
      t0 = 1. / ( s1 + s2 + s3 )
      s1 = s1 * t0
      s3 = s3 * t0
      ff(i,m) = ff(i,m) + ( s1*(t2-t1) + (0.5*s3-0.25)*(t3-t2) ) /3.
      enddo
      enddo

      enddo
*----------------- loop in "m"  ends  here  -------------------

* Project the fluxes to the physical space:

      do m =  1, mn
      do i = -1, ns
      fh(i,m) = evr(i,m,1) * ff(i,1) + evr(i,m,2) * ff(i,2)
     *+ evr(i,m,3) * ff(i,3) + evr(i,m,4) * ff(i,4)
     *+ ( -f(i-1,m) + 7*( f(i,m)+f(i+1,m) ) - f(i+2,m) )/12.
      enddo
      enddo

      return
      end


      subroutine rk(io)
      include 'comm.inc'
*****************************************************************
*     Runge-Kutta in time
*   mt = 3 ---> 3rd order Runge-Kutta (TVD)
*   mt = 4 ---> 4th order Runge-Kutta (non-TVD)
*****************************************************************

      if(mt.eq.3) then

      if(io.eq.0) then
      do m = 1, mn
      do j = 0, ny
      do i = 0, nx
      uc(i,j,m,1) = uc(i,j,m,0) + dt * rhs(i,j,m)
      enddo
      enddo
      enddo
      else if(io.eq.1) then
      do m = 1, mn
      do j = 0, ny
      do i = 0, nx
      uc(i,j,m,2) = 0.75 *   uc(i,j,m,0)
     *+ 0.25 * ( uc(i,j,m,1) + dt * rhs(i,j,m) )
      enddo
      enddo
      enddo
      else
      do m = 1, mn
      do j = 0, ny
      do i = 0, nx
      uc(i,j,m,0) = ( uc(i,j,m,0)
     *+ 2 * ( uc(i,j,m,2) + dt * rhs(i,j,m) ) )/3.
      enddo
      enddo
      enddo
      endif

      endif

      if(mt.eq.4) then

      if(io.le.1) then
      do m = 1, mn
      do j = 0, ny
      do i = 0, nx
      uc(i,j,m,io+1) = uc(i,j,m,0) + 0.5 * dt * rhs(i,j,m)
      enddo
      enddo
      enddo
      else if(io.eq.2) then
      do m = 1, mn
      do j = 0, ny
      do i = 0, nx
      uc(i,j,m,3) = uc(i,j,m,0) + dt * rhs(i,j,m)
      enddo
      enddo
      enddo
      else
      do m = 1, mn
      do j = 0, ny
      do i = 0, nx
      uc(i,j,m,0) = ( - uc(i,j,m,0) + uc(i,j,m,1)
     *+ 2 * uc(i,j,m,2) + uc(i,j,m,3)
     *+ 0.5 * dt * rhs(i,j,m) ) / 3.
      enddo
      enddo
      enddo
      endif

      endif

      return
      end


      subroutine save
      include 'comm.inc'
*****678****************************************************************
* Name:      save_per.f
* Function:  compute the L_1 & L_inf error in density variable
*****678****************************************************************

      write(*,*) ' '
      write(*,*) 'save the numerical solution'

      write(6,*) nt, ' t=',tnum
      write(9,*) 'zone i=',nx+1,', j=',ny+1

      ih=nx/2
      do 722 j=0,ny
      rij=uc(ih,j,2,0)/uc(ih,j,1,0)
      sij=uc(ih,j,3,0)/uc(ih,j,1,0)
      q2=rij**2+sij**2
      pij=(gamma-1.)*(uc(ih,j,4,0)-0.5*uc(ih,j,1,0)*q2)
      write(8,11) y(j),uc(ih,j,1,0),pij,rij,sij
      do 724 i=0,nx
      rij=uc(i,j,2,0)/uc(i,j,1,0)
      sij=uc(i,j,3,0)/uc(i,j,1,0)
      q2=rij**2+sij**2
      pij=(gamma-1.)*(uc(i,j,4,0)-0.5*uc(i,j,1,0)*q2)
      write(9,11) x(i),y(j),uc(i,j,1,0),pij,rij,sij
724   continue
722   continue
11    format(6e13.5)

      open(unit=10,file='restart',form='unformatted',status='unknown')
      write(10) tnum,(((uc(i,j,m,0),i=0,nx),j=0,ny),m=1,4)
      close(10)

      return
      end


      subroutine cflc(aam)

      include 'comm.inc'

c computed the largest eigenvalue for cfl

c     dimension w(nx,ny)

      aam=0.
      do 2 j=1,ny
      do 2 i=1,nx
      rij=uc(i,j,2,0)/uc(i,j,1,0)
      sij=uc(i,j,3,0)/uc(i,j,1,0)
      q2=rij**2+sij**2
      pij=(gamma-1.)*(uc(i,j,4,0)-0.5*uc(i,j,1,0)*q2)
c2=gamma*pij/uc(i,j,1,0)
cij=sqrt(abs(c2))

c     w(i,j)=(abs(rij)+cij)/dx+(abs(sij)+cij)/dy
      wtmp=(abs(rij)+cij)/dx+(abs(sij)+cij)/dy
      aam=max(aam,wtmp)
2     continue
c     nn=nx*ny
c     i0=ismax(nn,w(1,1),1)
c     aam=w(i0,1)

      return
      end
