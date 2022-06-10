function [all_pos_test_local, all_error_test, nets] = train_networks(pos_train, features_train, pos_test, features_test)
    all_box_data_train = split_into_boxes(pos_train, features_train);
    all_box_data_test = split_into_boxes(pos_test, features_test);
    all_error_test = [];
    all_pos_test_local = [];
    nets = {};
    for box_idx = 1:length(all_box_data_train)
        train = all_box_data_train{box_idx};
        test = all_box_data_test{box_idx};

        pos_box = train.pos;
        features_box = train.features;

        figure; hold on;
        scatter3(train.pos(:,1), train.pos(:,2), train.pos(:,3));
        scatter3(test.pos(:,1), test.pos(:,2), test.pos(:,3));

        fprintf("Box %d, Train: %d, Test: %d\n", box_idx, length(train.pos), length(test.pos));

        net = NNsolver(train.pos, train.features, [32], 0);

        local_pos_train_pred = sim(net, train.features')';
        local_pos_test_pred = sim(net, test.features')';
        dummy_test_pred = repmat(mean(test.pos), size(test.pos, 1), 1);

        error_train = local_pos_train_pred - train.pos;
        error_test = local_pos_test_pred - test.pos;
        error_dummy_test = dummy_test_pred - test.pos;

        mean_error_train = mean(sqrt(sum(error_train .^ 2, 2)));
        mean_error_test = mean(sqrt(sum(error_test .^ 2, 2)));
        mean_dummy_error_test = mean(sqrt(sum(error_dummy_test .^ 2, 2)));

        all_error_test = [all_error_test; error_test];
        all_pos_test_local = [all_pos_test_local; test.pos];
        nets{end+1} = net;
        fprintf("Mean train error (mm): %0.2f\n", mean_error_train);
        fprintf("Mean test error (mm): %0.2f\n", mean_error_test);
        fprintf("Baseline (mm): %0.2f\n", mean_dummy_error_test);
    end
end

