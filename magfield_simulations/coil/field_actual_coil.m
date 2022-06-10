function [B] = field_actual_coil(pos_coil, rx_pos)

BSmag = BSmag_init();

% parametrise transmitter coil: circular filament centred at origin
D_STEP = 1000;  % discretization increment [rad]

figure(); scatter3(pos_coil(:,1),pos_coil(:,2),pos_coil(:,3));


% add the transmitter coil; let I = 1 for M calculations
BSmag = BSmag_add_filament(BSmag, pos_coil, 1, 0.01/D_STEP);

% plot receiver sample locations
BSmag_plot_field_points(BSmag, rx_pos(:,1), rx_pos(:,2), rx_pos(:,3));

% calculate B at specified receiver location
[BSmag, X,Y,Z, BX, BY, BZ] = BSmag_get_B(BSmag, rx_pos(:,1), rx_pos(:,2), rx_pos(:,3));
B = [BX BY BZ]';

end

