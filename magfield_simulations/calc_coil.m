%% gives the resistance, inductance, and coil length of given wire gauge, radius, number of turns, number of layers, and magnetic permeability
%  wire gauge in standard AWG, radius in m
%  resistance in Ohms, inductance in H, coil length in m
function [RESISTANCE, INDUCTANCE, LENGTH] = calc_coil(AWG, radius, N, layers)
    switch AWG
        case 18
            R = 20.95;          % [mOhms / m]
            DIAMETER = 1.024;   % [mm]
        case 24
            R = 84.22;
            DIAMETER = 0.511;
        case 26
            R = 133.9;
            DIAMETER = 0.405;
        case 28
            R = 212.9;
            DIAMETER = 0.321;
        case 32
            R = 538.3;
            DIAMETER = 0.202;
    end
    
    RESISTANCE = R*10^-3 * (2*pi*radius) * N;         % [Ohms]
    
    LENGTH = N * DIAMETER*10^-3 / layers;             % [m]
    
    % INDUCTANCE = N^2 * (pi*radius^2) * mu / LENGTH;  % H
    % INDUCTANCE = N^2 * mu * radius * (log(16*radius./(DIAMETER*10^-3)) - 2); % [H]
     INDUCTANCE = 0.02 * N^2 * radius^2 / (6*radius + 9*LENGTH + 10*layers*DIAMETER);
    % INDUCTANCE = Lcalc(AWG, ceil(N / layers), layers, radius*2 ,f);
end