% Sigal Gottlieb September 7, 2003
% efficient WENO subroutine
%
% Modified June 22, 2007 by Daniel Higgs
% for use as a self-contained function
%
%
function rhs = weno(u, dx, f,fp)

% ------------- Begin Constants ----------------------

epsilon=10^(-29);		% Epsilon

md = 4;		% Number of ghost points to use
			% DON'T CHANGE THIS YET!
			% The consequences may be disasterous

% ------------- End Constants -------------------------


%  em = 1.0;
em = max(abs(fp(u)));

npoints = length(u);
nstart = md + 1;
np = npoints+md;



% It's possible (actually, it's intentended) that weno() may be passed
% vectors that contain many more points than the number of ghost points 
% being taken. Strip out all but the required number of ghost points as 
% defined by the variable md.
%

i = length(u);
%  uL = uL( i - md + 1: i);
%  uR = uR(1:md +1);


% Combine everything into a single vector
%  u = [ uL, uM, uR];
%  f = 1/2*u.^2;
%  f = inline('1/2*u.^2', 'u');

u = [ u(i-md:end-1), u, u(2:md+2)];


for i=nstart-md:np+md
    dfp(i)= (f(u(i+1))-f(u(i)) + em*(u(i+1) - u(i)))/2.0;
    dfm(i)= (f(u(i+1))-f(u(i)) - em*(u(i+1) - u(i)))/2.0;
end

for i = nstart-1:np+1
    hh(1,1) = dfp(i-2);
    hh(2,1) = dfp(i-1);
    hh(3,1) = dfp(i);
    hh(4,1) = dfp(i+1);
    hh(1,2) = - dfm(i+2);
    hh(2,2) = - dfm(i+1);
    hh(3,2) = - dfm(i);
    hh(4,2) = - dfm(i-1);
        
    fh(i)=0.0;
    fh(i) = ( -f(u(i-1)) + 7*(f(u(i))+f(u(i+1)))-f(u(i+2)) )/12.0;
    for m1=1:2
        t1 = hh(1,m1)-hh(2,m1);
        t2 = hh(2,m1)-hh(3,m1);
        t3 = hh(3,m1)-hh(4,m1);
        tt1=13.0*t1^2 + 3.0*(    hh(1,m1) - 3.0*hh(2,m1))^2;
        tt2=13.0*t2^2 + 3.0*(    hh(2,m1) +     hh(3,m1))^2;
        tt3=13.0*t3^2 + 3.0*(3.0*hh(3,m1) -     hh(4,m1))^2;
        tt1=(epsilon+tt1)^2;
        tt2=(epsilon+tt2)^2;
        tt3=(epsilon+tt3)^2;
        s1 = tt2*tt3;
        s2 = 6.0*tt1*tt3;
        s3 = 3.0*tt1*tt2;
        t0 = 1./(s1+s2+s3);
        s1 = s1*t0;
        s2 = s2*t0;
        s3 = s3*t0;
        fh(i)= fh(i) + (s1*(t2-t1) + (0.5*s3-0.25)*(t3-t2) )/3.0;
    end;
end;

for i = nstart:np
    rhs(i)= (fh(i-1)-fh(i) )/dx;
end;

% Only return the values of rhs that correspond to values of uM
% Strip away those that relate to the ghost points.
rhs = rhs(nstart:np);

end
