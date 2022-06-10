function [B] = field_dipole(tx_pos, tx_rot, rx_pos)
pos_world = rx_pos - tx_pos;
pos_coil = inv(tx_rot) * pos_world';

% x = r*sin(theta);
% y = r*sin(theta)*cos(phi);
% z = r*cos(theta);
x = pos_coil(1,:);
y = pos_coil(2,:);
z = pos_coil(3,:);

r = sqrt(x.^2 + y.^2 + z.^2);
Bx = 3*x.*z./(r.^5);
By = 3*y.*z./(r.^5);
Bz = (3*z.^2-r.^2)./(r.^5);
B = tx_rot * [Bx;By;Bz];

end

