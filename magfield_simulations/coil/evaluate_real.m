TRIAL = 't15';
processed = dlmread(['D:\mag_track\processed\processed__', TRIAL, '.csv'], ' ');
MAX_DIST = 1;
SPACE_DIV = 6;

% processed = processed(1:DECIMATE:1.48e4, :);

features = processed(:,1:6);
pos = processed(:,8:10);
rot = processed(:,11:14);

% features(:, 1:3) = acos(features(:,1:3));

scatter3(pos(:,1), pos(:,2), pos(:,3));
xlabel("X");
ylabel("Y");
zlabel("Z");

% TEST_START = 4.22e4;
% TEST_STOP = 5.0e4;
% TEST_START = 3.22e4/5;
% TEST_STOP = 4.0e4/5;
TEST_START = 5e4;
TEST_STOP = 6e4;

if TRIAL == 't15'
% t15
TEST_START = 7.3e4;
TEST_STOP = 9.6e4;
TEST_STOP = 8.5e4;
TEST_START = 8.5e4;
TEST_STOP = 9.5e4;
elseif TRIAL == 't16'
% t16
% TEST_START = 7.4e3;
% TEST_STOP = 9159;
TEST_START = 1.112e4;
TEST_STOP = 14340;

end

pos_train = [pos(1:TEST_START-1,:); pos(TEST_STOP:end,:)];
features_train = [features(1:TEST_START-1,:);features(TEST_STOP:end,:)];

pos_test = pos(TEST_START:TEST_STOP,:);
features_test = features(TEST_START:TEST_STOP,:);

figure; hold on;
scatter3(pos_train(:,1), pos_train(:,2), pos_train(:,3));
scatter3(pos_test(:,1), pos_test(:,2), pos_test(:,3));

[all_pos_test, all_error_test, nets] = train_networks(pos_train, features_train, pos_test, features_test);

run_networks(nets, features_test, pos_test);
error_mag = vecnorm(all_error_test, 2, 2);
fprintf("Mean test error (mm): %0.2f\n", mean(error_mag));
fprintf("RMS test error (mm): %0.2f\n", rms(error_mag));

distance = vecnorm(all_pos_test, 2, 2);

figure; scatter(distance, error_mag);

pos_test_pred = run_networks(nets, features_test, pos_test);
figure; hold on; plot(pos_test); plot(pos_test_pred);

data = [pos_test, pos_test_pred];
csvwrite(['D:\mag_track\processed\results__', TRIAL, '.csv'], data);
return ;
for space_count= 1:1
%     logic = pos(:,1) < 0 & vecnorm(pos,2,2) > (space_count)*MAX_DIST/SPACE_DIV & vecnorm(pos,2,2) < (space_count+1)*MAX_DIST/SPACE_DIV;
%     logic = distance > (space_count)*MAX_DIST/SPACE_DIV & distance < (space_count+1)*MAX_DIST/SPACE_DIV;
    logic_train = pos_train(:,1) > 50 & pos_train(:,1) < 150 & distance_train > 400 & distance_train < 600;
    logic_test = pos_test(:,1) > 50 & pos_test(:,1) < 150 & distance_test > 400 & distance_test < 600;
    logic_train = distance_train > 400 & distance_train < 500;
    logic_test = distance_test > 400 & distance_test < 500;
    logic = distance > 400 & distance < 500;
%     logic_train = distance_train < 800;
%     logic_test = distance_test < 800;
    
    local_pos_train = pos_train(logic_train,:,:);
    local_features_train = features_train(logic_train,:);
    
    local_pos_test = pos_test(logic_test,:,:);
    local_features_test = features_test(logic_test,:,:);
    
    local_pos = pos(logic,:,:);
    local_features = features(logic,:);
    
    fprintf("Train size: %d\n", size(local_pos_train, 1));
    fprintf("Test size: %d\n", size(local_pos_test, 1));
    
    figure; hold on;
    scatter3(local_pos_train(:,1), local_pos_train(:,2), local_pos_train(:,3), 3, '.');
    scatter3(local_pos_test(:,1), local_pos_test(:,2), local_pos_test(:,3), 3, '.');
    xlabel("X");
    ylabel("Y");
    zlabel("Z");
    
    
    figure;
    scatter3(local_pos_train(:,1), local_pos_train(:,2), local_pos_train(:,3), 3, local_features_train(:,1));
    xlabel('X'); ylabel('Y'); zlabel('Z');
    
    feature_mean = mean(local_features_train);
    feature_std = std(local_features_train);
    
    net = NNsolver(local_pos_train, (local_features_train - feature_mean) ./ feature_std);
    
    local_pos_train_pred = sim(net, ((local_features_train - feature_mean) ./ feature_std)')';
    local_pos_test_pred = sim(net, ((local_features_test - feature_mean) ./ feature_std)')';
    local_pos_pred = sim(net, ((local_features - feature_mean) ./ feature_std)')';
    dummy_test_pred = repmat(mean(local_pos_test), size(local_pos_test, 1), 1);
    
    error_train = local_pos_train_pred - local_pos_train;
    error_test = local_pos_test_pred - local_pos_test;
    error = local_pos_pred - local_pos;
    error_dummy_test = dummy_test_pred - local_pos_test;
    
    mean_error_train = mean(sqrt(sum(error_train .^ 2, 2)));
    mean_error_test = mean(sqrt(sum(error_test .^ 2, 2)));
    mean_dummy_error_test = mean(sqrt(sum(error_dummy_test .^ 2, 2)));
    
    fprintf("Mean train error (mm): %0.2f\n", mean_error_train);
    fprintf("Mean test error (mm): %0.2f\n", mean_error_test);
    fprintf("Baseline (mm): %0.2f\n", mean_dummy_error_test);
    
%     logic = pos(:,1) > 0 & vecnorm(pos,2,2) > (space_count)*MAX_DIST/SPACE_DIV & vecnorm(pos,2,2) < (space_count+1)*MAX_DIST/SPACE_DIV;
%     pos_temp = pos(logic,:,:);
%     disp(size(pos_temp));
%     features_pos_temp = features_pos(logic,:,:);
%     NNsolver(pos_temp, features_pos_temp);
end