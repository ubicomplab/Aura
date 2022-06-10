TRIAL = 't1';
jointAngle = dlmread(['D:\mag_ring\processed\jointAngle__', TRIAL, '.csv'], ' ');
mag = dlmread(['D:\mag_ring\processed\resampled__', TRIAL, '.csv'], ' ');


features = mag(:,1:6);

% scatter(jointAngle(:,1), jointAngle(:,2), jointAngle(:,3), jointAngle(:,4));

TEST_START = 3e4;
TEST_STOP = 4e4;


jointAngle_train = [jointAngle(1:TEST_START-1,:); jointAngle(TEST_STOP:end,:)];
features_train = [features(1:TEST_START-1,:);features(TEST_STOP:end,:)];

jointAngle_test = jointAngle(TEST_START:TEST_STOP,:);
features_test = features(TEST_START:TEST_STOP,:);


[all_jointAngle_test, all_error_test, nets] = train_networks(jointAngle_train, features_train, jointAngle_test, features_test);

run_networks(nets, features_test, jointAngle_test);
error_mag = vecnorm(all_error_test, 2, 2);
fprintf("Mean test error (mm): %0.2f\n", mean(error_mag));
fprintf("RMS test error (mm): %0.2f\n", rms(error_mag));

distance = vecnorm(all_jointAngle_test, 2, 2);

figure; scatter(distance, error_mag);

all_jointAngle_test_pred = run_networks(nets, features_test, jointAngle_test);
figure; hold on; plot(jointAngle_test); plot(all_jointAngle_test_pred);

data = [jointAngle_test, all_jointAngle_test_pred];
csvwrite(['D:\mag_ring\processed\results__', TRIAL, '.csv'], data);
return ;




