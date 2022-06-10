function [ M, TX_I, measurement, RX_V ] = GetMeasurements( Bfield, rx, tx, f, TX_V)
%GETMEASUREMENTS Summary of this function goes here
%   Detailed explanation goes here

theta = linspace(0, 2*pi, 360);
RX_U = [zeros(size(theta));    % receiver coil unit orientation vector
        sin(theta);
        cos(theta)];
    
RX_FLUX = Bfield.B' * RX_U * rx.area();    % magnetic flux when I = 1
M = abs(rx.n_turns_total * RX_FLUX) / 1;           % mutual inductance, note that this is divided by I so it all works out

omega = 2* pi*f;

Z1 = tx.R + tx.R_extra + 1i.*omega'*tx.L + 1./(1i.*omega'.*tx.C);
Z2 = rx.R + rx.R_extra + 1i.*omega'*rx.L + 1./(1i.*omega'.*rx.C);
ZM = 1i.*omega'*M;

TX_Z = Z1 - ZM.^2 ./ Z2;    % complex impedance
TX_I = TX_V ./ TX_Z;
TX_P = TX_I .* TX_V;


measurement = abs(TX_I * tx.R_extra);
RX_V0 = M;
RX_I = RX_V0 / (rx.R + 1j*(omega*rx.L - 1/(omega*rx.C)));
RX_V = RX_I / (1j * omega * rx.C);
end

