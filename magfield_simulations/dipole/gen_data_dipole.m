clear all;

N = 10^5; % number of points
degree = 0; % std angle in which the rx coils can rotate

x = ((rand([N,1])-.5) * 2 * .5) - 0;
y = ((rand([N,1])-.5) * 2 * .2) - .5;
z = ((rand([N,1])-.5) * 2 * .2) + .3;

rx_pos = [x, y, z];

% b1 = zeros(N,3);
% b2 = zeros(N,3);
% b3 = zeros(N,3);

tx_1_pos = [-.1,0,0];
tx_2_pos = [0.1,0,0];
tx_3_pos = [0,0,0];

tx_1 = rotx(15) * roty(45);
tx_2 = rotx(15) * roty(-45);
tx_3 = rotx(0);

b1 = field2(tx_1_pos, tx_1, rx_pos)';
b2 = field2(tx_2_pos, tx_2, rx_pos)';
b3 = field2(tx_3_pos, tx_3, rx_pos)';

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


m = [];
m = [m,abs(sum(b1 .* rx1_rot, 2))];
m = [m,abs(sum(b1 .* rx2_rot, 2))];
m = [m,abs(sum(b1 .* rx3_rot, 2))];
m = [m,abs(sum(b2 .* rx1_rot, 2))];
m = [m,abs(sum(b2 .* rx2_rot, 2))];
m = [m,abs(sum(b2 .* rx3_rot, 2))];
m = [m,abs(sum(b3 .* rx1_rot, 2))];
m = [m,abs(sum(b3 .* rx2_rot, 2))];
m = [m,abs(sum(b3 .* rx3_rot, 2))];

data = [m,x,y,z,qs];
% csvwrite('sim_dipole.csv', data);

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

data = [x,y,z,qs, m, ones(length(x), 1)];
header = 'x,y,z,qx,qy,qz,qw,m_0,m_1,m_2,m_3,m_4,m_5,m_6,m_7,m_8,is_valid';
file = 'D:\mag_track\processed\resampled_1rx_sim.csv';
fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header)
fclose(fid)
dlmwrite(file, data, '-append');

