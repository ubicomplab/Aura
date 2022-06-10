%% Produces a score based on the measured voltage across the transmitting coil
%  score related to inverse of measurement; lower score => larger measurement
function score = measure(x)
    TX_AWG = x(1);  % wire gauge
    RX_AWG = x(2);
    TX_D = x(3);    % coil diameter [m]
    RX_D = x(4);
    TX_NT = x(5);   % number of turns/layer
    RX_NT = x(6);
    TX_NL = x(7);   % number of layers
    RX_NL = x(8);
    TX_C = x(9);    % capacitance [F]
    RX_C = x(10);
    f = x(11);      % driving frequency of AC magnetic field [Hz]
    B = [x(12); x(13); x(14)] * (TX_NT*TX_NL);

    omega = 2*pi*f;
    TX_V = 5;

    % calculate coil characteristics
    [TX_L, TX_R, ~] = Lcalc(TX_AWG, TX_D, TX_NT, TX_NL, f); % [H, Ohms, mm]
    [RX_L, RX_R, ~] = Lcalc(RX_AWG, RX_D, RX_NT, RX_NL, f);
    
    % rotate receiver coil
    theta = linspace(0, pi/2, 2);
    RX_U = [zeros(1, size(theta,2));
            sin(theta);
            cos(theta)];
    
    % calculate flux & mutual inductance
    RX_FLUX = (B'*RX_U) * (pi*(RX_D/2)^2);
    M = RX_NT*RX_NL * RX_FLUX; % [H]

    % calculate current through transmitting coil
    Z1 = TX_R + 1i*omega*TX_L + 1/(1i*omega*TX_C);
    Z2 = RX_R + 1i*omega*RX_L + 1/(1i*omega*RX_C);
    ZM = 1i*omega.*M;
    TX_I = (TX_V*Z2) ./ (Z1*Z2 - ZM.^2);

    % calculate measurement voltage & score
    measurement = max(abs(TX_I * TX_R)) - min(abs(TX_I * TX_R));
    score = -measurement;
end