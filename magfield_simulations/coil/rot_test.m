data = dlmread('D:\mag_track\sim\sim.csv', ',', 1, 0);

% data = data(1:10000, :);

pos = data(:,1:3);
rot = data(:, 13:16);
distance = vecnorm(pos, 2, 2);

N = length(data);
tx1=data(:,17:19);
tx2=data(:,26:28);
tx3=data(:,35:37);
tx1_norot=data(:,44:46);
tx2_norot=data(:,53:55);
tx3_norot=data(:,62:64);
norot = [tx1_norot, tx2_norot, tx3_norot];
tx_rot = [tx1, tx2, tx3];

is_train = (1:N) < (N * .75);


pos_train = pos(is_train,:);
rot_train = rot(is_train,:);
norot_train = norot(is_train,:);
tx_rot_train = tx_rot(is_train,:);
distance_train = distance(is_train);

pos_test = pos(~is_train,:);
q_rot_test = rot(~is_train,:);
norot_test = norot(~is_train,:);
tx_rot_test = tx_rot(~is_train,:);
distance_test = distance(~is_train,:);





[tx_norot_pred, tx_rot_test_local, rot_test_local, norot_test_local, nets] = train_forward_networks(pos_train, norot_train, tx_rot_train, rot_train, pos_test, norot_test, tx_rot_test, q_rot_test);

tx1_norot=tx_norot_pred(:,1:3);
tx2_norot=tx_norot_pred(:,4:6);
tx3_norot=tx_norot_pred(:,7:9);
% tx1_norot=norot_test_local(:,1:3);
% tx2_norot=norot_test_local(:,4:6);
% tx3_norot=norot_test_local(:,7:9);

tx1_norot_norm = tx1_norot ./ vecnorm(tx1_norot, 2, 2);
tx2_norot_norm = tx2_norot ./ vecnorm(tx2_norot, 2, 2);
tx3_norot_norm = tx3_norot ./ vecnorm(tx3_norot, 2, 2);

tx1_norot_norm_gt = norot_test_local(:,1:3) ./ vecnorm(norot_test_local(:,1:3), 2, 2);
tx2_norot_norm_gt = norot_test_local(:,4:6) ./ vecnorm(norot_test_local(:,4:6), 2, 2);
tx3_norot_norm_gt = norot_test_local(:,7:9) ./ vecnorm(norot_test_local(:,7:9), 2, 2);

% figure;
% scatter3(pos_local(:,1), pos_local(:,2), pos_local(:,3), 1, tx3_norot(:,1));
% xlabel('X'); ylabel('Y'); zlabel('Z');

tx1_norm = tx_rot_test_local(:, 1:3) ./ vecnorm(tx_rot_test_local(:, 1:3), 2, 2);
tx2_norm = tx_rot_test_local(:, 4:6) ./ vecnorm(tx_rot_test_local(:, 4:6), 2, 2);
tx3_norm = tx_rot_test_local(:, 7:9) ./ vecnorm(tx_rot_test_local(:, 7:9), 2, 2);

errors = [];
for i = 1:length(tx1_norot)
    q = rot_test_local(i,:);
    
    H = tx1_norm(i,:)' * tx1_norot_norm(i,:) + tx2_norm(i,:)' * tx2_norot_norm(i,:) + tx3_norm(i,:)' * tx3_norot_norm(i,:);
    down_true = [0, -1, 0];
    down_hat = quatrotate((q), down_true);
    H = H + 1 * (down_true' * down_hat);
    [U,S,V] = svd(H);
    R = V * U';
    if det(R) < 0
%         disp("flip");
        V(:,3) = V(:,3) * -1;
        R = V * U'; 
%         R(:,3) = R(:,3) * -1;
    end
    q_hat = quatconj(rotm2quat(R));
    
%     disp(q_hat);
%     disp(q);
    error_q = quatmultiply(quatconj(q_hat), q);
    error = rad2deg(quat2angle(error_q));
    errors = [errors, error];
end
rms(errors)

csvwrite('D:\mag_track\processed\rot_gravity__sim.csv', errors);

errors = [];
for i = 1:length(tx1_norot)
    q = rot_test_local(i,:);
    H = tx1_norm(i,:)' * tx1_norot_norm(i,:) + tx2_norm(i,:)' * tx2_norot_norm(i,:) + tx3_norm(i,:)' * tx3_norot_norm(i,:);
    H = tx1_norm(i,:)' * tx1_norot_norm_gt(i,:) + tx2_norm(i,:)' * tx2_norot_norm_gt(i,:) + tx3_norm(i,:)' * tx3_norot_norm_gt(i,:);
    down_true = [0, -1, 0];
    down_hat = quatrotate((q), down_true);
%     H = H + 1 * (down_true' * down_hat);
    [U,S,V] = svd(H);
    R = V * U';
    if det(R) < 0
%         disp("flip");
        V(:,3) = V(:,3) * -1;
        R = V * U'; 
%         R(:,3) = R(:,3) * -1;
    end
    q_hat = quatconj(rotm2quat(R));
    
%     disp(q_hat);
%     disp(q);
    error_q = quatmultiply(quatconj(q_hat), q);
    error = rad2deg(quat2angle(error_q));
    errors = [errors, error];
end
rms(errors)



csvwrite('D:\mag_track\processed\rot_nogravity__sim.csv', errors);




