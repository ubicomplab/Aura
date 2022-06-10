function [pred, tx_rot, rot] = run_networks_rot(nets, pos_test, field_test, tx_rot_test, rot_test)
    all_box_data_test = split_into_boxes_rot(pos_test, field_test, tx_rot_test, rot_test);
    pred = zeros(length(pos_test), 9);
    rot = zeros(length(pos_test), 4);
    tx_rot = zeros(length(pos_test), 9);
    for box_idx = 1:length(all_box_data_test)
        test = all_box_data_test{box_idx};
        local_pos_test_pred = sim(nets{box_idx}, test.pos')';
        pred(test.indices,:) = local_pos_test_pred;
        rot(test.indices,:) = test.rot;
        tx_rot(test.indices,:) = test.tx_rot;
    end
end

