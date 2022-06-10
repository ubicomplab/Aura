clear all;
close all;
clc

points_coils = xlsread('yes');

points_coil1 = points_coils(1:200,:);
% points_coil2 = points_coils(201:400,:);
points_coil3 = points_coils(401:600,:);

figure();scatter3(points_coils(:,1),points_coils(:,2),points_coils(:,3));


N = 10^5; % number of points
degree = 50; % std angle in which the rx coils can rotate

x = ((rand([N,1])-.5) * 2 * .5) - 0;
y = ((rand([N,1])-.5) * 2 * .2) - .5;
z = ((rand([N,1])-.5) * 2 * .2) + .3;

rx_pos = [x, y, z];

tx_1_pos = sum(points_coil1,1)/size(points_coil1,1); radius_1 = [0.035,0.05];
tx_2_pos = sum(points_coil2,1)/size(points_coil2,1); radius_2 = [0.035,0.05];
tx_3_pos = sum(points_coil3,1)/size(points_coil3,1); radius_3 = [0.1,0.045];

tx_1 = rotx(15) * roty(45);
tx_2 = rotx(15) * roty(-45);
%reconfirm with Eric
tx_1 = rotx(15) * roty(-45);
tx_2 = rotx(15) * roty(45);
tx_3 = rotx(15);

b1_actual_coil= field_actual_coil(points_coil1, rx_pos)';
b2_actual_coil = field_actual_coil(points_coil2, rx_pos)';
b3_actual_coil = field_actual_coil(points_coil3, rx_pos)';

b1_coil_ellipse = field_coil_ellipse(tx_1_pos, tx_1, rx_pos,radius_1)';
b2_coi_ellipse = field_coil_ellipse(tx_2_pos, tx_2, rx_pos,radius_2)';
b3_coil_ellipse = field_coil_ellipse(tx_3_pos, tx_3, rx_pos,radius_3)';

b1_dipole = field_dipole(tx_1_pos, tx_1, rx_pos)';
b2_dipole = field_dipole(tx_2_pos, tx_2, rx_pos)';
b3_dipole = field_dipole(tx_3_pos, tx_3, rx_pos)';

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

m_coil = [];
m_coil = [m_coil,abs(sum(b1_coil_ellipse .* rx1_rot, 2))];
m_coil = [m_coil,abs(sum(b1_coil_ellipse .* rx2_rot, 2))];
m_coil = [m_coil,abs(sum(b1_coil_ellipse .* rx3_rot, 2))];
m_coil = [m_coil,abs(sum(b2_coi_ellipse .* rx1_rot, 2))];
m_coil = [m_coil,abs(sum(b2_coi_ellipse .* rx2_rot, 2))];
m_coil = [m_coil,abs(sum(b2_coi_ellipse .* rx3_rot, 2))];
m_coil = [m_coil,abs(sum(b3_coil_ellipse .* rx1_rot, 2))];
m_coil = [m_coil,abs(sum(b3_coil_ellipse .* rx2_rot, 2))];
m_coil = [m_coil,abs(sum(b3_coil_ellipse .* rx3_rot, 2))];

m_dipole = [];
m_dipole = [m_dipole,abs(sum(b1_dipole .* rx1_rot, 2))];
m_dipole = [m_dipole,abs(sum(b1_dipole .* rx2_rot, 2))];
m_dipole = [m_dipole,abs(sum(b1_dipole .* rx3_rot, 2))];
m_dipole = [m_dipole,abs(sum(b2_dipole .* rx1_rot, 2))];
m_dipole = [m_dipole,abs(sum(b2_dipole .* rx2_rot, 2))];
m_dipole = [m_dipole,abs(sum(b2_dipole .* rx3_rot, 2))];
m_dipole = [m_dipole,abs(sum(b3_dipole .* rx1_rot, 2))];
m_dipole = [m_dipole,abs(sum(b3_dipole .* rx2_rot, 2))];
m_dipole = [m_dipole,abs(sum(b3_dipole .* rx3_rot, 2))];

m_actual_coil = [];
m_actual_coil = [m_actual_coil,abs(sum(b1_actual_coil .* rx1_rot, 2))];
m_actual_coil = [m_actual_coil,abs(sum(b1_actual_coil .* rx2_rot, 2))];
m_actual_coil = [m_actual_coil,abs(sum(b1_actual_coil .* rx3_rot, 2))];
m_actual_coil = [m_actual_coil,abs(sum(b2_actual_coil .* rx1_rot, 2))];
m_actual_coil = [m_actual_coil,abs(sum(b2_actual_coil .* rx2_rot, 2))];
m_actual_coil = [m_actual_coil,abs(sum(b2_actual_coil .* rx3_rot, 2))];
m_actual_coil = [m_actual_coil,abs(sum(b3_actual_coil .* rx1_rot, 2))];
m_actual_coil = [m_actual_coil,abs(sum(b3_actual_coil .* rx2_rot, 2))];
m_actual_coil = [m_actual_coil,abs(sum(b3_actual_coil .* rx3_rot, 2))];

for i= (1:9)
    figure();
    scatter(m_coil(:,i),m_dipole(:,i));
end
for i= (1:9)
    figure();
    scatter(m_dipole(:,i),m_actual_coil(:,i));
end

data = [x,y,z,qs, m_dipole, m_actual_coil, ones(length(x), 1)];
header = 'x,y,z,qx,qy,qz,qw,md_0,md_1,md_2,md_3,md_4,md_5,md_6,md_7,md_8,m_0,m_1,m_2,m_3,m_4,m_5,m_6,m_7,m_8,is_valid';
file = 'D:\mag_track\processed\resampled_rot2_sim.csv';
fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header)
fclose(fid)
dlmwrite(file, data, '-append');
