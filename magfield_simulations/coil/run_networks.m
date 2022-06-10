function [pos_pred] = run_networks(nets, features_test, pos_test)
    all_box_data_test = split_into_boxes(pos_test, features_test);
    pos_pred = zeros(length(pos_test), 3);
    for box_idx = 1:length(all_box_data_test)
        test = all_box_data_test{box_idx};
        local_pos_test_pred = sim(nets{box_idx}, test.features')';
        pos_pred(test.indices,:) = local_pos_test_pred;
    end
end

