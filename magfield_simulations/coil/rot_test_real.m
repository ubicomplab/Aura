processed = dlmread('D:\mag_track\processed\processed__t15.csv', ' ');
features = processed(:,1:6);
pos = processed(:,8:10);
rot = processed(:,11:14);

norot = dlmread('C:\Users\emwhit\Dropbox\projects\mag-fingers\learning\preprocessing\norot_t15.txt', ' ');
distance = vecnorm(pos, 2, 2);
tx_rot = dlmread('C:\Users\emwhit\Dropbox\projects\mag-fingers\learning\preprocessing\rot_t15.txt', ' ');

TEST_START = 7.3e4;
TEST_STOP = 8.5e4;
TEST_START = 3e4;
TEST_STOP = 6e4;
TEST_START = 8.5e4;
TEST_STOP = 9.5e4;

pos_train = [pos(1:TEST_START-1,:); pos(TEST_STOP:end,:)];
rot_train = [rot(1:TEST_START-1,:);rot(TEST_STOP:end,:)];
norot_train = [norot(1:TEST_START-1,:); norot(TEST_STOP:end,:)];
tx_rot_train = [tx_rot(1:TEST_START-1,:);tx_rot(TEST_STOP:end,:)];
distance_train = [distance(1:TEST_START-1);distance(TEST_STOP:end)];

pos_test = pos(TEST_START:TEST_STOP,:);
rot_test = rot(TEST_START:TEST_STOP,:);
norot_test = norot(TEST_START:TEST_STOP,:);
tx_rot_test = tx_rot(TEST_START:TEST_STOP,:);
distance_test = distance(TEST_START:TEST_STOP,:);

indices = 1:length(distance);

[tx_norot_pred, tx_rot_test_local, rot_test_local, all_field_test, nets] = train_forward_networks(pos_train, norot_train, tx_rot_train, rot_train, pos_test, norot_test, tx_rot_test, rot_test);

[tx_norot_pred, tx_rot_test_local, rot_test_local] = run_networks_rot(nets, pos_test, norot_test, tx_rot_test, rot_test);

tx1_norot=tx_norot_pred(:,1:3);
tx2_norot=tx_norot_pred(:,4:6);
tx3_norot=tx_norot_pred(:,7:9);
tx1_norot_norm = tx1_norot ./ vecnorm(tx1_norot, 2, 2);
tx2_norot_norm = tx2_norot ./ vecnorm(tx2_norot, 2, 2);
tx3_norot_norm = tx3_norot ./ vecnorm(tx3_norot, 2, 2);


% figure;
% scatter3(pos_local(:,1), pos_local(:,2), pos_local(:,3), 1, tx3_norot(:,1));
% xlabel('X'); ylabel('Y'); zlabel('Z');

tx1_norm = tx_rot_test_local(:, 1:3) ./ vecnorm(tx_rot_test_local(:, 1:3), 2, 2);
tx2_norm = tx_rot_test_local(:, 4:6) ./ vecnorm(tx_rot_test_local(:, 4:6), 2, 2);
tx3_norm = tx_rot_test_local(:, 7:9) ./ vecnorm(tx_rot_test_local(:, 7:9), 2, 2);

valid_entries = sum(isnan(tx1_norot_norm), 2) == 0;

errors = [];
qs = [];
qs_hat = [];
for i = 1:length(tx1_norot)
    if ~valid_entries(i)
        continue
    end
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
    qs = [qs; q];
    qs_hat = [qs_hat; q_hat];
    
    
%     disp(q_hat);
%     disp(q);
    error_q = quatmultiply(quatconj(q_hat), q);
    error = rad2deg(quat2angle(error_q));
    errors = [errors, error];
end
rms(errors)

% csvwrite('D:\mag_track\processed\rot_gravity__t15.csv', errors);
csvwrite('D:\mag_track\processed\rot_gravity__t15_2.csv', [qs, qs_hat]);

qs = [];
qs_hat = [];
errors = [];
for i = 1:length(tx1_norot)
    if ~valid_entries(i)
        continue
    end
    q = rot_test_local(i,:);
    
    H = tx1_norm(i,:)' * tx1_norot_norm(i,:) + tx2_norm(i,:)' * tx2_norot_norm(i,:) + tx3_norm(i,:)' * tx3_norot_norm(i,:);
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
    qs = [qs; q];
    qs_hat = [qs_hat; q_hat];
    
    error_q = quatmultiply(quatconj(q_hat), q);
    error = rad2deg(quat2angle(error_q));
    errors = [errors, error];
end
rms(errors)
% csvwrite('D:\mag_track\processed\rot_nogravity__t15.csv', errors);
csvwrite('D:\mag_track\processed\rot_nogravity__t15_2.csv', [qs, qs_hat]);

% baseline = rot_test_local(2325,:);
% baseline = meanrot(rot_test_local);
baseline = mean(rot_test_local);
errors = [];
for i = 1:length(tx1_norot)
    if ~valid_entries(i)
        continue
    end
    q = rot_test_local(i,:);
    q_hat = baseline;
    
%     disp(q_hat);
%     disp(q);
    error_q = quatmultiply(quatconj(q_hat), q);
    error = rad2deg(quat2angle(error_q));
    errors = [errors, error];
end
rms(errors)
csvwrite('D:\mag_track\processed\rot_baseline__t15_2.csv', errors);