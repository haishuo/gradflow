%This code produces the Burgers' equation example 

clear all
%Burgers'
%f = @(u) -0.5*u^2; 
%fp = @(u) -u; 
%linear advection left moving
%f = @(u) (-u); 
%fp = @(u) (-1);
%linear advection right moving
f = @(u)  u; 
fp = @(u) 1;
%discretize space
x = linspace(-1, 1, 101);
%u0 = sin(pi*x); %sine wave IC
u0=sign(x); %step function IC
dx = max(diff(x));
h=0.5*dx; %what size is dt
u=u0;
N=75;
for j=1:N
   rhs=weno5(u0,dx,f,fp);
	u = u0 + h*rhs;
   rhs=weno5(u,dx,f,fp);
   u=0.75*u0+0.25*(u+h*rhs);
   rhs=weno5(u,dx,f,fp);
   u=(u0+2.0*(u+h*rhs))/3.0;
   u0=u;
   plot(u)
   pause (0.1)
end