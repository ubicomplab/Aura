RX_D = 20e-3;
TX_D = 200e-3;
RX_AWG = 26;
TX_AWG = 26;
TX_NT = 10;
RX_NT = 10;
N_LAYERS = 5;

N = 100000;

X = normrnd(0,.5,[N,1]);
Y = normrnd(-.5,.2,[N,1]);
Z = normrnd(.3,.2,[N,1]);

tx1 = Coil.FromWindings(TX_D, RX_NT, N_LAYERS, RX_AWG, TARGET_F, 0);
loc = [X,Y,Z];
Bfield1 = Field(loc, tx1, roty(30), [.2, 0, 0]);
Bfield2 = Field(loc, tx1, roty(-30), [-.2, 0, 0]);
Bfield3 = Field(loc, tx1, roty(0), [0, 0, 0]);
B1 = Bfield1.B;
B2 = Bfield2.B;
B3 = Bfield3.B;

figure; hold on;
scatter3(Bfield1.L(:,1), Bfield1.L(:,2), Bfield1.L(:,3),'.');
scatter3(Bfield2.L(:,1), Bfield2.L(:,2), Bfield2.L(:,3),'.');
scatter3(Bfield3.L(:,1), Bfield3.L(:,2), Bfield3.L(:,3),'.');
scatter3(X, Y, Z,'.');

rx1 = [1,0,0];
rx2 = [0,1,0];
rx3 = [0,0,1];

rX = normrnd(0,75,[N,1]);
rY = normrnd(0,75,[N,1]);
rZ = normrnd(0,75,[N,1]);

qs = zeros(N,4);
for i = 1:N
    M = rotx(rX(i)) * roty(rY(i)) * rotz(rZ(i));
    qs(i,:) = rotm2quat(M);
end

rx1_rot = quatrotate(qs, rx1);
rx2_rot = quatrotate(qs, rx2);
rx3_rot = quatrotate(qs, rx3);


m = [];
m = [m,sum(B1 .* rx1_rot, 2)];
m = [m,sum(B2 .* rx1_rot, 2)];
m = [m,sum(B3 .* rx1_rot, 2)];
m = [m,sum(B1 .* rx2_rot, 2)];
m = [m,sum(B2 .* rx2_rot, 2)];
m = [m,sum(B3 .* rx2_rot, 2)];
m = [m,sum(B1 .* rx3_rot, 2)];
m = [m,sum(B2 .* rx3_rot, 2)];
m = [m,sum(B3 .* rx3_rot, 2)];

data = [m,X,Y,Z,qs];
csvwrite('sim.csv', data);
