

wrist_len = 0.129516;
ring_len = 0.0415931;
rx_pos = [0.014916, 0.01, -0.020618];

direction_x = [1, 0, 0];
direction_y = [0, 1, 0];
direction_z = [0, 0, 1];


wrist_theta = jointAngle(:,1);
wrist_phi = jointAngle(:,2);
finger_theta = jointAngle(:,3);
finger_phi = jointAngle(:,4);
N = size(wrist_phi,1);

ring_pos = zeros(N, 3);
knuckle_pos = zeros(N, 3);
% rx_pos = zeros(N, 3);
% rx2_pos = zeros(N, 3);
% ring_frame = zeros(N, 3, 3);

for i = 1:N
    wrist_frame = rotz(wrist_theta(i)) * roty(wrist_phi(i));
    knuckle_pos(i, :) = (rx_pos' + wrist_frame * (direction_y * wrist_len)')';
    ring_frame_ = wrist_frame * rotz(finger_theta(i)) * roty(finger_phi(i));
    ring_pos(i, :) = (knuckle_pos(i,:)' + ring_frame_ * (direction_y * ring_len)')';
    
end

data = [knuckle_pos, ring_pos];
FILENAME = 'pred_from_jointAngle_t1';
file = ['D:\mag_ring\MATLAB\', FILENAME, '.csv'];
% fid = fopen(file,'w'); 
% fprintf(fid, header);
% fclose(fid);
dlmwrite(file, data, '-append');
