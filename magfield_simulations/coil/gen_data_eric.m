function [data, pos, rot, features_pos, features_rot] = gen_data_eric(N, space_count, use_cross)

if nargin <= 2
  use_cross = false;
end
addpath('field_models');
addpath('receiver_models');


USE_ACTUAL = 1;
USE_TRACE = 0;
USE_GEN_TRACE = 0;
TRACE = 't4';
FILENAME = 'sim_actual';
%FILENAME = 'eighthspace';
%N = 100000; % only used if USE_TRACE = 0
NUM_RX = 3;

if USE_ACTUAL
    points_coils = xlsread('yes');
    points_coil1 = points_coils(1:200,:);
    points_coil2 = points_coils(201:400,:);
    points_coil3 = points_coils(401:600,:);
    figure();scatter3(points_coils(:,1),points_coils(:,2),points_coils(:,3));

    tx1_pos = sum(points_coil1,1)/size(points_coil1,1); radius_1 = [0.035,0.05];
    tx2_pos = sum(points_coil2,1)/size(points_coil2,1); radius_2 = [0.035,0.05];
    tx3_pos = sum(points_coil3,1)/size(points_coil3,1); radius_3 = [0.1,0.045];

    tx1_rot = rotx(15) * roty(-45);
    tx2_rot = rotx(15) * roty(45);
    tx3_rot = rotx(15);

    % tx1 = DipoleModel(tx1_pos, tx1_rot);
    % tx2 = DipoleModel(tx2_pos, tx2_rot);
    % tx3 = DipoleModel(tx3_pos, tx3_rot);

    % tx1 = ellipseModel(tx1_pos, tx1_rot);
    % tx2 = ellipseModel(tx2_pos, tx2_rot);
    % tx3 = ellipseModel(tx3_pos, tx3_rot);

    tx1 = actualModel(points_coil1, tx1_rot);
    tx2 = actualModel(points_coil2, tx2_rot);
    tx3 = actualModel(points_coil3, tx3_rot);

    %model.Visualize();

%     N = 10^5;

else
%     tx1 = DipoleModel([0,0,0], rotx(0));
%     tx2 = DipoleModel([0,0,0], rotx(90));
%     tx3 = DipoleModel([0,0,0], roty(90));
    tx1 = DipoleModel([0,0,0], rotx(65));
    tx2 = DipoleModel([0,0,0], roty(80));
    tx3 = DipoleModel([0,0,0], roty(-80));
    %model.Visualize();

    eval_all = @(x) [tx1.Evaluate(x); tx2.Evaluate(x); tx3.Evaluate(x)];
    radius_1 = 0;
    radius_2 = 0;
    radius_3 = 0;
end

if NUM_RX == 1
    rx = Rx1();
elseif NUM_RX == 3
    rx = Rx3();
end

if use_cross
    x_v = linspace(-.7,.7, 100);
    z_v = linspace(0,.8, 100);
    [x,y,z] = meshgrid(x_v, [-.3], z_v);
    pos = [reshape(x, 1, []); reshape(y, 1, []); reshape(z, 1, [])]';
    rot = repmat([1,0,0,0], size(pos, 1), 1);
elseif USE_TRACE
%     [pos, rot] = load_trace(TRACE);
    if USE_GEN_TRACE
        [pos, rot] = load_gen_trace(TRACE);
        
    else
        [pos, rot] = load_new_trace(TRACE);
    end
    pos = pos(1:100000,:);
    rot = rot(1:100000,:);
    scatter3(pos(:,1),pos(:,2),pos(:,3));
    N = size(pos, 1);
else
    [pos, rot] = BaseRxModel.GenTransforms(N, space_count);
end

tx1_coil = zeros(size(pos,1), 3, 3);
tx2_coil = zeros(size(pos,1), 3, 3);
tx3_coil = zeros(size(pos,1), 3, 3);
disp("Starting tx 1");
tx1_coil(:,:,:) = rx.Measure(tx1, pos, rot, radius_1);
disp("Starting tx 2");
tx2_coil(:,:,:) = rx.Measure(tx2, pos, rot, radius_2);
disp("Starting tx 3");
tx3_coil(:,:,:) = rx.Measure(tx3, pos, rot,radius_3);


tx1_coil = reshape(tx1_coil, size(pos,1), []);
tx2_coil = reshape(tx2_coil, size(pos,1), []);
tx3_coil = reshape(tx3_coil, size(pos,1), []);

tx1_coil_no_rot = zeros(size(pos,1), 3, 3);
tx2_coil_no_rot = zeros(size(pos,1), 3, 3);
tx3_coil_no_rot = zeros(size(pos,1), 3, 3);
disp("Starting tx 1");
tx1_coil_no_rot(:,:,:) = rx.Measure(tx1, pos, [1,0,0,0], radius_1);
disp("Starting tx 2");
tx2_coil_no_rot(:,:,:) = rx.Measure(tx2, pos, [1,0,0,0], radius_2);
disp("Starting tx 3");
tx3_coil_no_rot(:,:,:) = rx.Measure(tx3, pos, [1,0,0,0], radius_3);


tx1_coil_no_rot = reshape(tx1_coil_no_rot, size(pos,1), []);
tx2_coil_no_rot = reshape(tx2_coil_no_rot, size(pos,1), []);
tx3_coil_no_rot = reshape(tx3_coil_no_rot, size(pos,1), []);

coilPos = rx.GenCoilPositions(pos, rot);
if size(coilPos, 3) < 3
    coilPos(end, end, 3) = 0;
end
pos = pos * 1000;
coilPos1 = coilPos(:,:,1) * 1000;
coilPos2 = coilPos(:,:,2) * 1000;
coilPos3 = coilPos(:,:,3) * 1000;

tx1 = tx1_coil(:,1:3);
tx2 = tx2_coil(:,1:3);
tx3 = tx3_coil(:,1:3);

% dot_products = [dot(tx1, tx2, 2), dot(tx1, tx3, 2), dot(tx2, tx3, 2)];

dot_products = abs([dot(tx1, tx2, 2)./(vecnorm(tx1, 2, 2).*vecnorm(tx2, 2, 2)), dot(tx1, tx3, 2)./(vecnorm(tx1, 2, 2).*vecnorm(tx3, 2, 2)), dot(tx2, tx3, 2)./(vecnorm(tx2, 2, 2).*vecnorm(tx3, 2, 2))]);
magnitudes = [vecnorm(tx1, 2, 2), vecnorm(tx2, 2, 2), vecnorm(tx3, 2, 2)];
dot_cross = [dot(tx1, cross(tx2, tx3), 2)];

if use_cross
figure;imagesc([-700,700],[0,800],reshape(dot_products(:,2), 100, 100)');
xlabel("X (mm)");
ylabel("Z (mm)");
colorbar
figure;imagesc([-700,700],[0,800],reshape(magnitudes(:,2), 100, 100)');
xlabel("X (mm)");
ylabel("Z (mm)");
colorbar
end
features_rot = [dot_products, tx1, tx2, tx3];
features_pos = [dot_products, magnitudes];
data = [pos, coilPos1, coilPos2, coilPos3, rot, tx1_coil, tx2_coil, tx3_coil, tx1_coil_no_rot, tx2_coil_no_rot, tx3_coil_no_rot, features_pos, features_rot];
%header = 'x,y,z,x1,y1,z1,x2,y2,z2,x3,y3,z3,qw,qx,qy,qz,t1r1x,t1r1y,t1r1z,t2r1x,t2r1y,t2r1z,t3r1x,t3r1y,t3r1z,is_valid';
% [tx1_coil(:,1:3),tx2_coil(:,1:3),tx3_coil(:,1:3)]
header = 'x,y,z,';
header = [header, 'x1,y1,z1,x2,y2,z2,x3,y3,z3,'];
header = [header, 'qw,qx,qy,qz,'];
header = [header, 't1r1x,t1r1y,t1r1z,t1r2x,t1r2y,t1r2z,t1r3x,t1r3y,t1r3z,t2r1x,t2r1y,t2r1z,t2r2x,t2r2y,t2r2z,t2r3x,t2r3y,t2r3z,t3r1x,t3r1y,t3r1z,t3r2x,t3r2y,t3r2z,t3r3x,t3r3y,t3r3z,'];
header = [header, 't1r1x_no_rot,t1r1y_no_rot,t1r1z_no_rot,t1r2x_no_rot,t1r2y_no_rot,t1r2z_no_rot,t1r3x_no_rot,t1r3y_no_rot,t1r3z_no_rot,t2r1x_no_rot,t2r1y_no_rot,t2r1z_no_rot,t2r2x_no_rot,t2r2y_no_rot,t2r2z_no_rot,t2r3x_no_rot,t2r3y_no_rot,t2r3z_no_rot,t3r1x_no_rot,t3r1y_no_rot,t3r1z_no_rot,t3r2x_no_rot,t3r2y_no_rot,t3r2z_no_rot,t3r3x_no_rot,t3r3y_no_rot,t3r3z_no_rot,d12,d13,d23,m1,m2,m3'];
file = ['D:\mag_track\sim\', FILENAME, '.csv'];

fid = fopen(file,'w'); 
fprintf(fid,'%s\n',header);
fclose(fid);
dlmwrite(file, data, '-append');
end