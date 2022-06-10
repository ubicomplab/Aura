processed = dlmread('D:\mag_track\processed\processed__t16.csv', ' ');
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


% t16
TEST_START = 7.4e3;
TEST_STOP = 9633;

pos_train = [pos(1:TEST_START-1,:); pos(TEST_STOP:end,:)];
features_train = [features(1:TEST_START-1,:);features(TEST_STOP:end,:)];

pos_test = pos(TEST_START:TEST_STOP,:);
features_test = features(TEST_START:TEST_STOP,:);

figure; hold on;
scatter3(pos_train(:,1), pos_train(:,2), pos_train(:,3));
scatter3(pos_test(:,1), pos_test(:,2), pos_test(:,3));

[all_pos_test, all_error_test, nets] = train_networks(pos_train, features_train, pos_test, features_test);

error_mag = vecnorm(all_error_test, 2, 2);
fprintf("Mean test error (mm): %0.2f\n", mean(error_mag));
fprintf("RMS test error (mm): %0.2f\n", rms(error_mag));

distance = vecnorm(all_pos_test, 2, 2);

figure; scatter(distance, error_mag);

processed = dlmread('D:\mag_track\processed\processed__t16_precision.csv', ' ');
MAX_DIST = 1;
SPACE_DIV = 6;

% processed = processed(1:DECIMATE:1.48e4, :);

features = processed(:,1:6);
pos = processed(:,8:10);
rot = processed(:,11:14);

boxes = split_into_boxes(pos, features);

extractions = { [750, 1600, 2300, 3300], [400, 1000], [400, 1200], []};
              
all_deviations = [];
for i = 1:4
    pos_pred = sim(nets{i}, boxes{i}.features')';
    regions = extractions{i};
    figure; plot(pos_pred);
    figure; hold on;
    for j = 1:length(regions)
        region = regions(j);
        pos_region = pos_pred(region:region+300,:);
        figure; plot(pos_region);
        center = mean(pos_region);
        deviations = vecnorm(pos_region - center, 2, 2);
        all_deviations = [all_deviations; deviations];
    end
    figure; cdfplot(all_deviations)
end

csvwrite('D:\mag_track\processed\deviations__t16_precision.csv', all_deviations);

