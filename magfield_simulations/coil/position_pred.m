TEST_START = 8000;
TEST_STOP = 10000;


% [~, pos, rot, features_pos, features_rot] = gen_data_eric(100000, 0);
data = dlmread('D:\mag_track\sim\sim_actual.csv', ',', 1, 0);
% data = dlmread('D:\mag_track\sim\trace_t4.csv', ',', 1, 0);
indices = 1:length(data);
is_train = (indices < TEST_START) | (indices >= TEST_STOP);

pos_train = data(is_train,1:3);
features_train = data(is_train, 71:76);
pos_test = data(~is_train,1:3);
features_test = data(~is_train, 71:76);

[all_pos_test, all_error_test] = train_networks(pos_train, features_train, pos_test, features_test);

error_mag = vecnorm(all_error_test, 2, 2);
fprintf("Mean test error (mm): %0.2f\n", mean(error_mag));
fprintf("RMS test error (mm): %0.2f\n", rms(error_mag));

distance = vecnorm(all_pos_test, 2, 2);

figure; scatter(distance, error_mag);


% csvwrite('D:\mag_track\processed\pos_sim.csv', error_mag);
csvwrite('D:\mag_track\processed\pos_sim_actual.csv', error_mag);