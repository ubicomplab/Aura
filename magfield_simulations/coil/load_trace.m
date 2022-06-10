function [pos, quat] = load_trace(name)
    traces = readtable('D:\mag_track\recordings\recordings.txt', 'ReadVariableNames', true);
    cell = traces(strcmp(traces.key, name), 'file_id');
    file_id = cell{1,1}{1};
    
    filepath = fullfile('D:\mag_track\recordings', ['combo_', file_id, '.txt']);
    
    data = csvread(filepath);
    % time = data(:,1);
    % sensors_raw = data(:,2:10);
    % sensors_filt = data(:,11:19);
    head_opti = data(:,20:26);
    hand_opti = data(:,27:33);
    
    head_pos = head_opti(:,1:3);
    head_q = head_opti(:,4:end);
    head_q = head_q(:,[4,1,2,3]);
    hand_pos = hand_opti(:,1:3);
    hand_q = hand_opti(:,4:end);
    hand_q = hand_q(:,[4,1,2,3]);
    
    hand_rel_pos_world = hand_pos - head_pos;
    hand_rel_pos_head = quatrotate(head_q, hand_rel_pos_world);
    
    hand_q_hand_to_head = quatmultiply(hand_q, quatconj(head_q));
    % TODO: Check all of this
    
    figure; 
    plot(hand_rel_pos_head);
    
    pos = hand_rel_pos_head;
    quat = hand_q_hand_to_head;
end

