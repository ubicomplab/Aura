clear all;
close all;
clc

points_coils = xlsread('yes');

points_coil1 = points_coils(1:200,:);
points_coil2 = points_coils(201:400,:);
points_coil3 = points_coils(401:600,:);

figure();scatter3(points_coils(:,1),points_coils(:,2),points_coils(:,3));


N = 10^3; % number of points
degree = 50; % std angle in which the rx coils can rotate

x1 = ((rand([N,1])-.5) * 2 * .5) - 0;
y1 = ((rand([N,1])-.5) * 2 * .2) - .5;
z1 = ((rand([N,1])-.5) * 2 * .2) + .3;

x2 = x1 - 0.1;%+ 0.02;%+0.02;
y2 = y1 + 0.03;%+ 0.01;%+0.01
z2 = z1 + 0.1;%+ 0.02;%+0.04

x3 = x1 + 0.05; %+ 0.04;%+0.04
y3  = y1 - 0.05;%+ 0.02;%+0.03
z3 = z1 + 0.06; %+ 0.02;%+0.02

rx_pos1 = [x1, y1, z1];
rx_pos2 = [x2, y2, z2];
rx_pos3 = [x3, y3, z3];

tx_1_pos = sum(points_coil1,1)/size(points_coil1,1); radius_1 = [0.035,0.05];
tx_2_pos = sum(points_coil2,1)/size(points_coil2,1); radius_2 = [0.035,0.05];
tx_3_pos = sum(points_coil3,1)/size(points_coil3,1); radius_3 = [0.1,0.045];

tx_1 = rotx(15) * roty(45);
tx_2 = rotx(15) * roty(-45);
%reconfirm with Eric
tx_1 = rotx(15) * roty(-45);
tx_2 = rotx(15) * roty(45);
tx_3 = rotx(15);

% b11_actual_coil = field_actual_coil(points_coil1, rx_pos1)';
% b21_actual_coil = field_actual_coil(points_coil2, rx_pos1)';
% b31_actual_coil = field_actual_coil(points_coil3, rx_pos1)';
% 
% b12_actual_coil = field_actual_coil(points_coil1, rx_pos2)';
% b22_actual_coil = field_actual_coil(points_coil2, rx_pos2)';
% b32_actual_coil = field_actual_coil(points_coil3, rx_pos2)';
% 
% b13_actual_coil = field_actual_coil(points_coil1, rx_pos3)';
% b23_actual_coil = field_actual_coil(points_coil2, rx_pos3)';
% b33_actual_coil = field_actual_coil(points_coil3, rx_pos3)';

b11_coil_ellipse = field_coil_ellipse(tx_1_pos, tx_1, rx_pos1,radius_1)';
b21_coi_ellipse = field_coil_ellipse(tx_2_pos, tx_2, rx_pos1,radius_1)';
b31_coil_ellipse = field_coil_ellipse(tx_3_pos, tx_3, rx_pos1,radius_1)';

b12_coil_ellipse = field_coil_ellipse(tx_1_pos, tx_1, rx_pos2,radius_2)';
b22_coi_ellipse = field_coil_ellipse(tx_2_pos, tx_2, rx_pos2,radius_2)';
b32_coil_ellipse = field_coil_ellipse(tx_3_pos, tx_3, rx_pos2,radius_2)';

b13_coil_ellipse = field_coil_ellipse(tx_1_pos, tx_1, rx_pos3,radius_3)';
b23_coi_ellipse = field_coil_ellipse(tx_2_pos, tx_2, rx_pos3,radius_3)';
b33_coil_ellipse = field_coil_ellipse(tx_3_pos, tx_3, rx_pos3,radius_3)';

% b11_dipole = field_dipole(tx_1_pos, tx_1, rx_pos1)';
% b21_dipole = field_dipole(tx_2_pos, tx_2, rx_pos1)';
% b31_dipole = field_dipole(tx_3_pos, tx_3, rx_pos1)';
% 
% b12_dipole = field_dipole(tx_1_pos, tx_1, rx_pos2)';
% b22_dipole = field_dipole(tx_2_pos, tx_2, rx_pos2)';
% b32_dipole = field_dipole(tx_3_pos, tx_3, rx_pos2)';
% 
% b13_dipole = field_dipole(tx_1_pos, tx_1, rx_pos3)';
% b23_dipole = field_dipole(tx_2_pos, tx_2, rx_pos3)';
% b33_dipole = field_dipole(tx_3_pos, tx_3, rx_pos3)';

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

m1_coil = [];
m1_coil = [m1_coil,(sum(b11_coil_ellipse .* rx1_rot, 2))];
m1_coil = [m1_coil,(sum(b11_coil_ellipse .* rx2_rot, 2))];
m1_coil = [m1_coil,(sum(b11_coil_ellipse .* rx3_rot, 2))];
m1_coil = [m1_coil,(sum(b21_coi_ellipse .* rx1_rot, 2))];
m1_coil = [m1_coil,(sum(b21_coi_ellipse .* rx2_rot, 2))];
m1_coil = [m1_coil,(sum(b21_coi_ellipse .* rx3_rot, 2))];
m1_coil = [m1_coil,(sum(b31_coil_ellipse .* rx1_rot, 2))];
m1_coil = [m1_coil,(sum(b31_coil_ellipse .* rx2_rot, 2))];
m1_coil = [m1_coil,(sum(b31_coil_ellipse .* rx3_rot, 2))];

m2_coil = [];
m2_coil = [m2_coil,(sum(b12_coil_ellipse .* rx1_rot, 2))];
m2_coil = [m2_coil,(sum(b12_coil_ellipse .* rx2_rot, 2))];
m2_coil = [m2_coil,(sum(b12_coil_ellipse .* rx3_rot, 2))];
m2_coil = [m2_coil,(sum(b22_coi_ellipse .* rx1_rot, 2))];
m2_coil = [m2_coil,(sum(b22_coi_ellipse .* rx2_rot, 2))];
m2_coil = [m2_coil,(sum(b22_coi_ellipse .* rx3_rot, 2))];
m2_coil = [m2_coil,(sum(b32_coil_ellipse .* rx1_rot, 2))];
m2_coil = [m2_coil,(sum(b32_coil_ellipse .* rx2_rot, 2))];
m2_coil = [m2_coil,(sum(b32_coil_ellipse .* rx3_rot, 2))];

m3_coil = [];
m3_coil = [m3_coil,(sum(b13_coil_ellipse .* rx1_rot, 2))];
m3_coil = [m3_coil,(sum(b13_coil_ellipse .* rx2_rot, 2))];
m3_coil = [m3_coil,(sum(b13_coil_ellipse .* rx3_rot, 2))];
m3_coil = [m3_coil,(sum(b23_coi_ellipse .* rx1_rot, 2))];
m3_coil = [m3_coil,(sum(b23_coi_ellipse .* rx2_rot, 2))];
m3_coil = [m3_coil,(sum(b23_coi_ellipse .* rx3_rot, 2))];
m3_coil = [m3_coil,(sum(b33_coil_ellipse .* rx1_rot, 2))];
m3_coil = [m3_coil,(sum(b33_coil_ellipse .* rx2_rot, 2))];
m3_coil = [m3_coil,(sum(b33_coil_ellipse .* rx3_rot, 2))];

% m1_dipole = [];
% m1_dipole = [m1_dipole,abs(sum(b11_dipole .* rx1_rot, 2))];
% m1_dipole = [m1_dipole,abs(sum(b11_dipole .* rx2_rot, 2))];
% m1_dipole = [m1_dipole,abs(sum(b11_dipole .* rx3_rot, 2))];
% m1_dipole = [m1_dipole,abs(sum(b21_dipole .* rx1_rot, 2))];
% m1_dipole = [m1_dipole,abs(sum(b21_dipole .* rx2_rot, 2))];
% m1_dipole = [m1_dipole,abs(sum(b21_dipole .* rx3_rot, 2))];
% m1_dipole = [m1_dipole,abs(sum(b31_dipole .* rx1_rot, 2))];
% m1_dipole = [m1_dipole,abs(sum(b31_dipole .* rx2_rot, 2))];
% m1_dipole = [m1_dipole,abs(sum(b31_dipole .* rx3_rot, 2))];

% m2_dipole = [];
% m2_dipole = [m2_dipole,abs(sum(b12_dipole .* rx1_rot, 2))];
% m2_dipole = [m2_dipole,abs(sum(b12_dipole .* rx2_rot, 2))];
% m2_dipole = [m2_dipole,abs(sum(b12_dipole .* rx3_rot, 2))];
% m2_dipole = [m2_dipole,abs(sum(b22_dipole .* rx1_rot, 2))];
% m2_dipole = [m2_dipole,abs(sum(b22_dipole .* rx2_rot, 2))];
% m2_dipole = [m2_dipole,abs(sum(b22_dipole .* rx3_rot, 2))];
% m2_dipole = [m2_dipole,abs(sum(b32_dipole .* rx1_rot, 2))];
% m2_dipole = [m2_dipole,abs(sum(b32_dipole .* rx2_rot, 2))];
% m2_dipole = [m2_dipole,abs(sum(b32_dipole .* rx3_rot, 2))];

% m3_dipole = [];
% m3_dipole = [m3_dipole,abs(sum(b13_dipole .* rx1_rot, 2))];
% m3_dipole = [m3_dipole,abs(sum(b13_dipole .* rx2_rot, 2))];
% m3_dipole = [m3_dipole,abs(sum(b13_dipole .* rx3_rot, 2))];
% m3_dipole = [m3_dipole,abs(sum(b23_dipole .* rx1_rot, 2))];
% m3_dipole = [m3_dipole,abs(sum(b23_dipole .* rx2_rot, 2))];
% m3_dipole = [m3_dipole,abs(sum(b23_dipole .* rx3_rot, 2))];
% m3_dipole = [m3_dipole,abs(sum(b33_dipole .* rx1_rot, 2))];
% m3_dipole = [m3_dipole,abs(sum(b33_dipole .* rx2_rot, 2))];
% m3_dipole = [m3_dipole,abs(sum(b33_dipole .* rx3_rot, 2))];

% m1_actual_coil = [];
% m1_actual_coil = [m1_actual_coil,(sum(b11_actual_coil .* rx1_rot, 2))];
% m1_actual_coil = [m1_actual_coil,(sum(b11_actual_coil .* rx2_rot, 2))];
% m1_actual_coil = [m1_actual_coil,(sum(b11_actual_coil .* rx3_rot, 2))];
% m1_actual_coil = [m1_actual_coil,(sum(b21_actual_coil .* rx1_rot, 2))];
% m1_actual_coil = [m1_actual_coil,(sum(b21_actual_coil .* rx2_rot, 2))];
% m1_actual_coil = [m1_actual_coil,(sum(b21_actual_coil .* rx3_rot, 2))];
% m1_actual_coil = [m1_actual_coil,(sum(b31_actual_coil .* rx1_rot, 2))];
% m1_actual_coil = [m1_actual_coil,(sum(b31_actual_coil .* rx2_rot, 2))];
% m1_actual_coil = [m1_actual_coil,(sum(b31_actual_coil .* rx3_rot, 2))];

% m2_actual_coil = [];
% m2_actual_coil = [m2_actual_coil,(sum(b12_actual_coil .* rx1_rot, 2))];
% m2_actual_coil = [m2_actual_coil,(sum(b12_actual_coil .* rx2_rot, 2))];
% m2_actual_coil = [m2_actual_coil,(sum(b12_actual_coil .* rx3_rot, 2))];
% m2_actual_coil = [m2_actual_coil,(sum(b22_actual_coil .* rx1_rot, 2))];
% m2_actual_coil = [m2_actual_coil,(sum(b22_actual_coil .* rx2_rot, 2))];
% m2_actual_coil = [m2_actual_coil,(sum(b22_actual_coil .* rx3_rot, 2))];
% m2_actual_coil = [m2_actual_coil,(sum(b32_actual_coil .* rx1_rot, 2))];
% m2_actual_coil = [m2_actual_coil,(sum(b32_actual_coil .* rx2_rot, 2))];
% m2_actual_coil = [m2_actual_coil,(sum(b32_actual_coil .* rx3_rot, 2))];

% m3_actual_coil = [];
% m3_actual_coil = [m3_actual_coil,(sum(b13_actual_coil .* rx1_rot, 2))];
% m3_actual_coil = [m3_actual_coil,(sum(b13_actual_coil .* rx2_rot, 2))];
% m3_actual_coil = [m3_actual_coil,(sum(b13_actual_coil .* rx3_rot, 2))];
% m3_actual_coil = [m3_actual_coil,(sum(b23_actual_coil .* rx1_rot, 2))];
% m3_actual_coil = [m3_actual_coil,(sum(b23_actual_coil .* rx2_rot, 2))];
% m3_actual_coil = [m3_actual_coil,(sum(b23_actual_coil .* rx3_rot, 2))];
% m3_actual_coil = [m3_actual_coil,(sum(b33_actual_coil .* rx1_rot, 2))];
% m3_actual_coil = [m3_actual_coil,(sum(b33_actual_coil .* rx2_rot, 2))];
% m3_actual_coil = [m3_actual_coil,(sum(b33_actual_coil .* rx3_rot, 2))];

% for i= (1:9)
%     figure();
%     scatter(m_coil(:,i),m_dipole(:,i));
% end
for i= (1:9)
    figure();
    scatter3(m1_coil(:,i),m2_coil(:,i),m3_coil(:,i));
end

data2 = [x1, y1, z1, x1,y1,z1,x2,y2,z2,qs, m1_coil, m2_coil, ones(length(x1), 1)];
header = 'x,y,z,x1,y1,z1,x2,y2,z2,qx,qy,qz,qw,m1_0,m1_1,m1_2,m1_3,m1_4,m1_5,m1_6,m1_7,m1_8,m2_0,m2_1,m2_2,m2_3,m2_4,m2_5,m2_6,m2_7,m2_8,is_valid';
file = 'D:\mag_track\processed\resampled_3tx_2rx_simEllipsis_noABS.csv';
fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header)
fclose(fid)
dlmwrite(file, data2, '-append');

data = [x1, y1, z1, x1,y1,z1,x2,y2,z2,x3,y3,z3,qs, m1_coil, m2_coil, m3_coil, ones(length(x1), 1)];
header = 'x,y,z,x1,y1,z1,x2,y2,z2,x3,y3,z3,qx,qy,qz,qw,t1r1x,t1r1y,t1r1z,t1r2x,t1r2y,t1r2z,t1r3x,t1r3y,t1r3z,t2r1x,t2r1y,t2r1z,t2r2x,t2r2y,t2r2z,t2r3x,t2r3y,t2r3z,t3r1x,t3r1y,t3r1z,t3r2x,t3r2y,t3r2z,t3r3x,t3r3y,t3r3z,is_valid';
%header = 'x,y,z,x1,y1,z1,x2,y2,z2,x3,y3,z3,qx,qy,qz,qw,m1_0,m1_1,m1_2,m1_3,m1_4,m1_5,m1_6,m1_7,m1_8,m2_0,m2_1,m2_2,m2_3,m2_4,m2_5,m2_6,m2_7,m2_8,m3_0,m3_1,m3_2,m3_3,m3_4,m3_5,m3_6,m3_7,m3_8,is_valid';
file = 'D:\mag_track\processed\resampled_3tx_3rx_simEllipsis_noABS.csv';
fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header)
fclose(fid)
dlmwrite(file, data, '-append');

data3 = [x1, y1, z1, x1,y1,z1,qs, m1_coil, ones(length(x1), 1)];
header = 'x,y,z,x1,y1,z1,qx,qy,qz,qw,m1_0,m1_1,m1_2,m1_3,m1_4,m1_5,m1_6,m1_7,m1_8,is_valid';
file = 'D:\mag_track\processed\resampled_3tx_1rx_simEllipsis_noABS.csv';
fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header)
fclose(fid)
dlmwrite(file, data2, '-append');
