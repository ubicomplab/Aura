function [all_field_pred_test, all_txrot_test, all_rot_test, all_field_test, nets] = train_forward_networks(pos_train, field_train, tx_rot_train, rot_train, pos_test, field_test, tx_rot_test, rot_test)
    all_box_data_train = split_into_boxes_rot(pos_train, field_train, tx_rot_train, rot_train);
    all_box_data_test = split_into_boxes_rot(pos_test, field_test, tx_rot_test, rot_test);
    all_field_pred_test = [];
    all_txrot_test = [];
    all_rot_test = [];
    all_field_test = [];
    nets = {};
    for box_idx = 1:length(all_box_data_train)
        train = all_box_data_train{box_idx};
        test = all_box_data_test{box_idx};

        figure; hold on;
        scatter3(train.pos(:,1), train.pos(:,2), train.pos(:,3));
        scatter3(test.pos(:,1), test.pos(:,2), test.pos(:,3));

        fprintf("Box %d, Train: %d, Test: %d\n", box_idx, length(train.pos), length(test.pos));

        net = NNsolver(train.field, train.pos, [32], 0);

        local_pos_test_pred = sim(net, test.pos')';

        all_field_pred_test = [all_field_pred_test; local_pos_test_pred];
        all_txrot_test = [all_txrot_test; test.tx_rot];
        all_rot_test = [all_rot_test; test.rot];
        all_field_test = [all_field_test; test.field];
        nets{end+1} = net;
        
    end
end

