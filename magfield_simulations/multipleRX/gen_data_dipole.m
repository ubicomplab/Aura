clear all;

N = 10^5; % number of points
degree = 0; % std angle in which the rx coils can rotate

x1 = ((rand([N,1])-.5) * 2 * .5) - 0;
y1 = ((rand([N,1])-.5) * 2 * .2) - .5;
z1 = ((rand([N,1])-.5) * 2 * .2) + .3;


x2 = x1;
y2 = y1;
z2 = z1 + 0.02;

x3 = x1;
y3 = y1;
z3 = z1 + 0.04;

rx_pos1 = [x1, y1, z1];
rx_pos2 = [x2, y2, z2];
rx_pos3 = [x3, y3, z3];


tx_1_pos = [-.1,0,0];
tx_2_pos = [0.1,0,0];
tx_3_pos = [0,0,0];

tx_1 = rotx(15) * roty(-45);
tx_2 = rotx(15) * roty(45);
tx_3 = rotx(15);

b11 = field2(tx_1_pos, tx_1, rx_pos1)';
b21 = field2(tx_2_pos, tx_2, rx_pos1)';
b31 = field2(tx_3_pos, tx_3, rx_pos1)';

b12 = field2(tx_1_pos, tx_1, rx_pos2)';
b22 = field2(tx_2_pos, tx_2, rx_pos2)';
b32 = field2(tx_3_pos, tx_3, rx_pos2)';

b13 = field2(tx_1_pos, tx_1, rx_pos3)';
b23 = field2(tx_2_pos, tx_2, rx_pos3)';
b33 = field2(tx_3_pos, tx_3, rx_pos3)';

% for i = 1:N
%     b1(i,:) = field([x(i,1),y(i,1),z(i,1)]);
%     b2(i,:) = (roty(-30)*(field((roty(30)*[x(i,1)-0.2;y(i,1);z(i,1)+0.05]).')).').';
%     b3(i,:) = (roty(30)*(field((roty(-30)*[x(i,1)+0.2;y(i,1);z(i,1)+0.05]).')).').';
% end

rx1 = [1,0,0];
rx2 = [0,1,0];
rx3 = [0,0,1];

rX = normrnd(0,degree,[N,1]);
rY = normrnd(0,degree,[N,1]);
rZ = normrnd(0,degree,[N,1]);

qs = zeros(N,4);
for i = 1:N
    M = rotx(rX(i)) * roty(rY(i)) * rotz(rZ(i));
    qs(i,:) = rotm2quat(M);
end

rx1_rot = quatrotate(qs, rx1);
rx2_rot = quatrotate(qs, rx2);
rx3_rot = quatrotate(qs, rx3);


m1 = [];
m1 = [m1,abs(sum(b11 .* rx1_rot, 2))];
m1 = [m1,abs(sum(b11 .* rx2_rot, 2))];
m1 = [m1,abs(sum(b11 .* rx3_rot, 2))];
m1 = [m1,abs(sum(b21 .* rx1_rot, 2))];
m1 = [m1,abs(sum(b21 .* rx2_rot, 2))];
m1 = [m1,abs(sum(b21 .* rx3_rot, 2))];
m1 = [m1,abs(sum(b31 .* rx1_rot, 2))];
m1 = [m1,abs(sum(b31 .* rx2_rot, 2))];
m1 = [m1,abs(sum(b31 .* rx3_rot, 2))];


m2 = [];
m2 = [m2,abs(sum(b12 .* rx1_rot, 2))];
m2 = [m2,abs(sum(b12 .* rx2_rot, 2))];
m2 = [m2,abs(sum(b12 .* rx3_rot, 2))];
m2 = [m2,abs(sum(b22 .* rx1_rot, 2))];
m2 = [m2,abs(sum(b22 .* rx2_rot, 2))];
m2 = [m2,abs(sum(b22 .* rx3_rot, 2))];
m2 = [m2,abs(sum(b32 .* rx1_rot, 2))];
m2 = [m2,abs(sum(b32 .* rx2_rot, 2))];
m2 = [m2,abs(sum(b32 .* rx3_rot, 2))];

m3 = [];
m3 = [m3,abs(sum(b13 .* rx1_rot, 2))];
m3 = [m3,abs(sum(b13 .* rx2_rot, 2))];
m3 = [m3,abs(sum(b13 .* rx3_rot, 2))];
m3 = [m3,abs(sum(b23 .* rx1_rot, 2))];
m3 = [m3,abs(sum(b23 .* rx2_rot, 2))];
m3 = [m3,abs(sum(b23 .* rx3_rot, 2))];
m3 = [m3,abs(sum(b33 .* rx1_rot, 2))];
m3 = [m3,abs(sum(b33 .* rx2_rot, 2))];
m3 = [m3,abs(sum(b33 .* rx3_rot, 2))];


% cross talk on tx and rx
USE_CROSS = 0;
if USE_CROSS
    upThresh = 0.5;
    lowThresh = 1;
    crossTalk_tx = (upThresh-lowThresh).*rand(1,3) + lowThresh;
    % crossTalk_tx = [1,1,1];

    b1_tx_cross = b1.*crossTalk_tx(1) + b2.*((1-crossTalk_tx(1))/2) + b3.*((1-crossTalk_tx(1))/2);
    b2_tx_cross = b1.*((1-crossTalk_tx(2))/2) + b2.*crossTalk_tx(2) + b3.*((1-crossTalk_tx(2))/2);
    b3_tx_cross = b1.*((1-crossTalk_tx(3))/2) + b2.*((1-crossTalk_tx(3))/2) + b3.*crossTalk_tx(3);


    crossTalk_rx = (upThresh-lowThresh).*rand(1,3) + lowThresh;
    % crossTalk_rx = [1,1,1];
    c11 = sum(b1_tx_cross .* rx1_rot, 2);
    c21 = sum(b2_tx_cross .* rx1_rot, 2);
    c31 = sum(b3_tx_cross .* rx1_rot, 2);
    c12 = sum(b1_tx_cross .* rx2_rot, 2);
    c22 = sum(b2_tx_cross .* rx2_rot, 2);
    c32 = sum(b3_tx_cross .* rx2_rot, 2);
    c13 = sum(b1_tx_cross .* rx3_rot, 2);
    c23 = sum(b2_tx_cross .* rx3_rot, 2);
    c33 = sum(b3_tx_cross .* rx3_rot, 2);

    c11_cross = c11.*crossTalk_rx(1) + c21.*((1-crossTalk_rx(1))/2) + c31.*((1-crossTalk_rx(1))/2);
    c21_cross = c11.*((1-crossTalk_rx(1))/2)+ c21.*crossTalk_rx(1) + c31.*((1-crossTalk_rx(1))/2);
    c31_cross = c11.*((1-crossTalk_rx(1))/2) + c21.*((1-crossTalk_rx(1))/2) + c31.*crossTalk_rx(1);

    c12_cross = c12.*crossTalk_rx(2) + c22.*((1-crossTalk_rx(2))/2) + c32.*((1-crossTalk_rx(2))/2);
    c22_cross = c12.*((1-crossTalk_rx(2))/2)+ c22.*crossTalk_rx(2) + c32.*((1-crossTalk_rx(2))/2);
    c32_cross = c12.*((1-crossTalk_rx(2))/2) + c22.*((1-crossTalk_rx(2))/2) + c32.*crossTalk_rx(2);

    c13_cross = c13.*crossTalk_rx(3) + c23.*((1-crossTalk_rx(3))/2) + c33.*((1-crossTalk_rx(3))/2);
    c23_cross = c13.*((1-crossTalk_rx(3))/2) + c23.*crossTalk_rx(3) + c33.*((1-crossTalk_rx(3))/2);
    c33_cross = c13.*((1-crossTalk_rx(3))/2) + c23.*((1-crossTalk_rx(3))/2) + c33.*crossTalk_rx(3);


    m_cross = [];
    m_cross = [m_cross,abs(c11_cross)];
    m_cross = [m_cross,abs(c12_cross)];
    m_cross = [m_cross,abs(c13_cross)];
    m_cross = [m_cross,abs(c21_cross)];
    m_cross = [m_cross,abs(c22_cross)];
    m_cross = [m_cross,abs(c23_cross)];
    m_cross = [m_cross,abs(c31_cross)];
    m_cross = [m_cross,abs(c32_cross)];
    m_cross = [m_cross,abs(c33_cross)];

    time = ((0:N-1)/120)';

    data = [m_cross,x,y,z,qs];
    csvwrite('sim_cross_dipole.csv', data);

    qs2=circshift(qs, -1, 2);
    data = [time,m_cross,m_cross,zeros(N,6),ones(N,1),x,y,z,qs2];
    % csvwrite('sim.csv', data);
end

data = [x1,y1,z1,x2,y2,z2,x3,y3,z3,qs, m1, m2, m3, ones(length(x1), 1)];
header = 'x1,y1,z1,x2,y2,z2,x3,y3,z3,qx,qy,qz,qw,m1_0,m1_1,m1_2,m1_3,m1_4,m1_5,m1_6,m1_7,m1_8,m2_0,m2_1,m2_2,m2_3,m2_4,m2_5,m2_6,m2_7,m2_8,m3_0,m3_1,m3_2,m3_3,m3_4,m3_5,m3_6,m3_7,m3_8,is_valid';
file = 'D:\mag_track\processed\resampled_3rx_sim.csv';
fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header)
fclose(fid)
dlmwrite(file, data, '-append');

