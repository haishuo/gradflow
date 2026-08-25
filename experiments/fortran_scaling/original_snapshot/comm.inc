*****678****************************************************************       
*                      comm.inc
*	   	nsd = max( nxd, nyd )
* dx,dy,dt are step sizes in x,y and t.
* uc: the numerical solution
* rhs: represents all the stuff on the right hand side of the ODE
*      in time, i.e.  u_t = rhs
*****678****************************************************************       

      parameter( md=3, mnm=4 )
      parameter( nxd=200+md, nyd=200+md, nsd=200+md )

* constants:
      common/const_int/ntot,mt,mn,nx,nxm,ny,nym,nt
      common/const_real/gamma,gm1,cfl,epweno,tend

* grid related variables:
      common/grid/dx,cdx,x(-md:nxd)
     *             ,dy,cdy,y(-md:nyd)
     *             ,dt,em,tnum,velocity

* necessary vars for solutions:      
      common/core_2D/uc(-md:nxd,-md:nyd,mnm,0:3)
     *                ,rhs(-md:nxd,-md:nyd,mnm)

* temporary 1D arrays
      common/temp_1D/u(-md:nsd,mnm), f(-md:nsd,mnm)
     *                ,evl(-md:nsd,mnm,mnm), evr(-md:nsd,mnm,mnm)
     *                ,am(mnm), fh(-md:nsd,mnm)
