clear all;
close all;
clc

points_coils = xlsread('yes');

points_coil1 = points_coils(1:200,:);
% points_coil2 = points_coils(201:400,:);
% points_coil3 = points_coils(401:600,:);

% figure();scatter3(points_coils(:,1),points_coils(:,2),points_coils(:,3));


N = 10^5; % number of points
degree = 50; % std angle in which the rx coils can rotate

x1 = ((rand([N,1])-.5) * 2 * .5) - 0;
y1 = ((rand([N,1])-.5) * 2 * .2) - .5;
z1 = ((rand([N,1])-.5) * 2 * .2) + .3;

x2 = x1 + 0.02;
y2 = y1 + 0.01;
z2 = z1 + 0.04;

x3 = x1 + 0.04;
y3  = y1 + 0.03;
z3 = z1 + 0.02;

rx_pos1 = [x1, y1, z1];
rx_pos2 = [x2, y2, z2];
rx_pos3 = [x3, y3, z3];

tx_1_pos = sum(points_coil1,1)/size(points_coil1,1); radius_1 = [0.035,0.05];
% tx_2_pos = sum(points_coil2,1)/size(points_coil2,1); radius_2 = [0.035,0.05];
% tx_3_pos = sum(points_coil3,1)/size(points_coil3,1); radius_3 = [0.1,0.045];

tx_1 = rotx(15) * roty(45);
% tx_2 = rotx(15) * roty(-45);
%reconfirm with Eric
tx_1 = rotx(15) * roty(-45);
% tx_2 = rotx(15) * roty(45);
% tx_3 = rotx(15);

b11_actual_coil = field_actual_coil(points_coil1, rx_pos1)';
% b21_actual_coil = field_actual_coil(points_coil2, rx_pos1)';
% b31_actual_coil = field_actual_coil(points_coil3, rx_pos1)';

b12_actual_coil = field_actual_coil(points_coil1, rx_pos2)';
% b22_actual_coil = field_actual_coil(points_coil2, rx_pos2)';
% b32_actual_coil = field_actual_coil(points_coil3, rx_pos2)';

b13_actual_coil = field_actual_coil(points_coil1, rx_pos3)';
% b23_actual_coil = field_actual_coil(points_coil2, rx_pos3)';
% b33_actual_coil = field_actual_coil(points_coil3, rx_pos3)';

% b1_coil_ellipse = field_coil_ellipse(tx_1_pos, tx_1, rx_pos,radius_1)';
% b2_coi_ellipse = field_coil_ellipse(tx_2_pos, tx_2, rx_pos,radius_2)';
% b3_coil_ellipse = field_coil_ellipse(tx_3_pos, tx_3, rx_pos,radius_3)';
% 
% b1_dipole = field_dipole(tx_1_pos, tx_1, rx_pos)';
% b2_dipole = field_dipole(tx_2_pos, tx_2, rx_pos)';
% b3_dipole = field_dipole(tx_3_pos, tx_3, rx_pos)';

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

% m_coil = [];
% m_coil = [m_coil,abs(sum(b1_coil_ellipse .* rx1_rot, 2))];
% m_coil = [m_coil,abs(sum(b1_coil_ellipse .* rx2_rot, 2))];
% m_coil = [m_coil,abs(sum(b1_coil_ellipse .* rx3_rot, 2))];
% m_coil = [m_coil,abs(sum(b2_coi_ellipse .* rx1_rot, 2))];
% m_coil = [m_coil,abs(sum(b2_coi_ellipse .* rx2_rot, 2))];
% m_coil = [m_coil,abs(sum(b2_coi_ellipse .* rx3_rot, 2))];
% m_coil = [m_coil,abs(sum(b3_coil_ellipse .* rx1_rot, 2))];
% m_coil = [m_coil,abs(sum(b3_coil_ellipse .* rx2_rot, 2))];
% m_coil = [m_coil,abs(sum(b3_coil_ellipse .* rx3_rot, 2))];
% 
% m_dipole = [];
% m_dipole = [m_dipole,abs(sum(b1_dipole .* rx1_rot, 2))];
% m_dipole = [m_dipole,abs(sum(b1_dipole .* rx2_rot, 2))];
% m_dipole = [m_dipole,abs(sum(b1_dipole .* rx3_rot, 2))];
% m_dipole = [m_dipole,abs(sum(b2_dipole .* rx1_rot, 2))];
% m_dipole = [m_dipole,abs(sum(b2_dipole .* rx2_rot, 2))];
% m_dipole = [m_dipole,abs(sum(b2_dipole .* rx3_rot, 2))];
% m_dipole = [m_dipole,abs(sum(b3_dipole .* rx1_rot, 2))];
% m_dipole = [m_dipole,abs(sum(b3_dipole .* rx2_rot, 2))];
% m_dipole = [m_dipole,abs(sum(b3_dipole .* rx3_rot, 2))];

m1_actual_coil = [];
m1_actual_coil = [m1_actual_coil,abs(sum(b11_actual_coil .* rx1_rot, 2))];
m1_actual_coil = [m1_actual_coil,abs(sum(b11_actual_coil .* rx2_rot, 2))];
m1_actual_coil = [m1_actual_coil,abs(sum(b11_actual_coil .* rx3_rot, 2))];
% m1_actual_coil = [m1_actual_coil,abs(sum(b21_actual_coil .* rx1_rot, 2))];
% m1_actual_coil = [m1_actual_coil,abs(sum(b21_actual_coil .* rx2_rot, 2))];
% m1_actual_coil = [m1_actual_coil,abs(sum(b21_actual_coil .* rx3_rot, 2))];
% m1_actual_coil = [m1_actual_coil,abs(sum(b31_actual_coil .* rx1_rot, 2))];
% m1_actual_coil = [m1_actual_coil,abs(sum(b31_actual_coil .* rx2_rot, 2))];
% m1_actual_coil = [m1_actual_coil,abs(sum(b31_actual_coil .* rx3_rot, 2))];

m2_actual_coil = [];
m2_actual_coil = [m2_actual_coil,abs(sum(b12_actual_coil .* rx1_rot, 2))];
m2_actual_coil = [m2_actual_coil,abs(sum(b12_actual_coil .* rx2_rot, 2))];
m2_actual_coil = [m2_actual_coil,abs(sum(b12_actual_coil .* rx3_rot, 2))];
% m2_actual_coil = [m2_actual_coil,abs(sum(b22_actual_coil .* rx1_rot, 2))];
% m2_actual_coil = [m2_actual_coil,abs(sum(b22_actual_coil .* rx2_rot, 2))];
% m2_actual_coil = [m2_actual_coil,abs(sum(b22_actual_coil .* rx3_rot, 2))];
% m2_actual_coil = [m2_actual_coil,abs(sum(b32_actual_coil .* rx1_rot, 2))];
% m2_actual_coil = [m2_actual_coil,abs(sum(b32_actual_coil .* rx2_rot, 2))];
% m2_actual_coil = [m2_actual_coil,abs(sum(b32_actual_coil .* rx3_rot, 2))];

m3_actual_coil = [];
m3_actual_coil = [m3_actual_coil,abs(sum(b13_actual_coil .* rx1_rot, 2))];
m3_actual_coil = [m3_actual_coil,abs(sum(b13_actual_coil .* rx2_rot, 2))];
m3_actual_coil = [m3_actual_coil,abs(sum(b13_actual_coil .* rx3_rot, 2))];
% m3_actual_coil = [m3_actual_coil,abs(sum(b23_actual_coil .* rx1_rot, 2))];
% m3_actual_coil = [m3_actual_coil,abs(sum(b23_actual_coil .* rx2_rot, 2))];
% m3_actual_coil = [m3_actual_coil,abs(sum(b23_actual_coil .* rx3_rot, 2))];
% m3_actual_coil = [m3_actual_coil,abs(sum(b33_actual_coil .* rx1_rot, 2))];
% m3_actual_coil = [m3_actual_coil,abs(sum(b33_actual_coil .* rx2_rot, 2))];
% m3_actual_coil = [m3_actual_coil,abs(sum(b33_actual_coil .* rx3_rot, 2))];

% for i= (1:9)
%     figure();
%     scatter(m_coil(:,i),m_dipole(:,i));
% end
for i= (1:9)
    figure();
    scatter3(m1_actual_coil(:,i),m2_actual_coil(:,i),m3_actual_coil(:,i));
end

data2 = [x1,y1,z1,x2,y2,z2,qs, m1_actual_coil, m2_actual_coil, ones(length(x1), 1)];
header = 'x1,y1,z1,x2,y2,z2,qx,qy,qz,qw,m1_0,m1_1,m1_2,m2_0,m2_1,m2_2,is_valid';
file = 'D:\mag_track\processed\resampled_1tx_18rx_simCoil.csv';
fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header)
fclose(fid)
dlmwrite(file, data2, '-append');

data = [x1,y1,z1,x2,y2,z2,x3,y3,z3,qs, m1_actual_coil, m2_actual_coil, m3_actual_coil, ones(length(x1), 1)];
header = 'x1,y1,z1,x2,y2,z2,x3,y3,z3,qx,qy,qz,qw,m1_0,m1_1,m1_2,m2_0,m2_1,m2_2,m3_0,m3_1,m3_2,is_valid';
file = 'D:\mag_track\processed\resampled_1tx_27rx_simCoil.csv';
fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header)
fclose(fid)
dlmwrite(file, data, '-append');

data3 = [x1,y1,z1,qs, m1_actual_coil, m2_actual_coil, ones(length(x1), 1)];
header = 'x1,y1,z1,qx,qy,qz,qw,m1_0,m1_1,m1_2,is_valid';
file = 'D:\mag_track\processed\resampled_1tx_9rx_simCoil.csv';
fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header)
fclose(fid)
dlmwrite(file, data2, '-append');
