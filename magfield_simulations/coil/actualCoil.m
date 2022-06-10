clear all;
%close all;
clc

coil3_base = xlsread('coil3-base');
coil3_all = xlsread('coil3-all');
coil1_base = xlsread('coil1-base');
coil1_all = xlsread('coil1-all');

coil_1_3_all = xlsread('coil1-3-all');

points_coil3 = coil3_all(size(coil3_base,1)+1:size(coil3_all,1),:);
points_coil1 = coil1_all(size(coil1_base,1)+1:size(coil1_all,1),:);
points_coil2 = (roty(180) * points_coil1')';

all_data_norot = vertcat(points_coil1,points_coil3,points_coil2);
figure();scatter(all_data_norot(:,1),all_data_norot(:,2));
figure();scatter(all_data_norot(:,1),all_data_norot(:,2));

tx_1_x = rotx(-15);
tx_2_x = rotx(-15);
tx_3_x = rotx(-15);
tx_1_y = roty(-45);
tx_2_y = roty(45);
tx_3_y = roty(0);

coil1 = (tx_1_y*points_coil1')';
coil2 = (tx_2_y*points_coil2')';
coil3 = (tx_3_y*points_coil3')';
coil1_2 = (tx_1_x*coil1')';
coil2_2 = (tx_2_x*coil2')';
coil3_2 = (tx_3_x*coil3')';
all_data = vertcat(coil1,coil2, coil3);
all_data_2 = vertcat(coil1_2,coil2_2, coil3_2);
figure();scatter3(all_data(:,1),all_data(:,2),all_data(:,3));
figure();scatter3(all_data_2(:,1),all_data_2(:,2),all_data_2(:,3));


N = 10^3; % number of points
degree = 0; % std angle in which the rx coils can rotate

x = ((rand([N,1])-.5) * 2 * .5) - 0;
y = ((rand([N,1])-.5) * 2 * .2) - .5;
z = ((rand([N,1])-.5) * 2 * .2) + .3;

rx_pos = [x, y, z];

tx_1_x = roty(45)* rotx(15);
tx_2_x = rotx(15) * roty(-45);
tx_3_x = rotx(15);

pos_coil1 = (tx_1_x) * points_coil1';
%pos_coil2 = (tx_2) * points_coil2';
pos_coil3 = (tx_3_x) * points_coil3';

BSmag_1 = BSmag_init();
BSmag_3 = BSmag_init();

% parametrise transmitter coil: circular filament centred at origin
D_STEP = 1000;  % discretization increment [rad]

figure(); scatter3(pos_coil1(1,:),pos_coil1(2,:),pos_coil1(3,:));
figure(); scatter3(pos_coil3(1,:),pos_coil3(2,:),pos_coil3(3,:));

% add the transmitter coil; let I = 1 for M calculations
BSmag_1 = BSmag_add_filament(BSmag_1, pos_coil1', 1, 0.01/D_STEP);
BSmag_3 = BSmag_add_filament(BSmag_3, pos_coil3', 1, 0.01/D_STEP);


% plot receiver sample locations
% BSmag_plot_field_points(BSmag, pos_coil(1,:), pos_coil(2,:), pos_coil(3,:));

% calculate B at specified receiver location
[BSmag_1, X,Y,Z, BX_1, BY_1, BZ_1] = BSmag_get_B(BSmag_1, rx_pos(:,1), rx_pos(:,2), rx_pos(:,3));
[BSmag_3, X,Y,Z, BX_3, BY_3, BZ_3] = BSmag_get_B(BSmag_3, rx_pos(:,1), rx_pos(:,2), rx_pos(:,3));

B_1 = (inv(tx_1_x) *[BX_1 BY_1 BZ_1]')';
B_3 = (inv(tx_3_x) *[BX_3 BY_3 BZ_3]')';