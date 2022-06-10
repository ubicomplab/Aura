%% Calculates internal inductance (Henries/meter) using formula Li-ACA3.8ML by D. Knight
%  s = ratio of skin depth to wire diameter
%  return value is Henries/m
function [L] =  LiCorr(s)
    z = 1 / (5.104*s);
    mu = 4*pi*1e-7;
    y = 0.0239 / (1 + 1.67*(z^0.036 - z^-0.72)^2)^4;
    L = mu/2/pi * s * (1 - exp(-(0.25/s)^3.8))^(1/3.8) * (1-y) - 5e-8;
end