%% Calculate the outside diameter of magnet wire insulation based on the conductor diameter id.
function [od] = odCalc(id)
    m = 0.96344;
    b = -0.19861;
    od = max(exp(m*log(id) + b), 1.09*id);
end