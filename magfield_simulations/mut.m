%% Calculate mutual inductance in microhenries between two coaxial circular filaments
%  r1,r2 are radii of the respective loops in cm
%  x is the axial distance separating them in cm
function [m] = mut(r1,r2,x)
      muo = pi*4e-7;
      a = sqrt((r1+r2)^2 + x^2);
      b = sqrt((r1-r2)*(r1-r2) + x^2);
      c = a - b;
      ci = 1;
      cs = c*c;
      co = c + 1;
      while (c<co)
        ao = (a+b) / 2;
        b = sqrt(a*b);
        a = ao;
        co = c;
        c = a - b;
        ci = 2*ci;
        cs = cs + ci*c^2;
      end
      m = muo/8 * pi * cs/a;
  end