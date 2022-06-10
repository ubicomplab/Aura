TX_D = .08; TX_N = 500; RX_X = 0; RX_Y = 0; RX_Z = .12;
RX_X = linspace(-1,1,50); RX_Y = linspace(-1,1,50); RX_Z = linspace(-1,1,50);

BSmag = BSmag_init();

% parametrise transmitter coil: circular filament centred at origin
D_STEP = 1000;  % discretization increment [rad]
a = linspace(0, 2*pi, D_STEP);
L = zeros(length(a),3);
for i = 1:length(a)
    L(i,:) = (TX_D/2) * [cos(a(i)), sin(a(i)), 0];
end

% add the transmitter coil; let I = 1 for M calculations
BSmag = BSmag_add_filament(BSmag, L, 1, 1/D_STEP);

% plot receiver sample locations
BSmag_plot_field_points(BSmag, RX_X, RX_Y, RX_Z);

% calculate B at specified receiver location
[BSmag, X,Y,Z, BX, BY, BZ] = BSmag_get_B(BSmag, RX_X, RX_Y, RX_Z);
B = [BX; BY; BZ] * TX_N