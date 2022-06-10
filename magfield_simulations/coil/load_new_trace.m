function [pos, quat] = load_new_trace(name)
    traces = readtable('D:\mag_track\recordings_native\recordings.txt', 'ReadVariableNames', true, 'Delimiter', ',');
    cell = traces(strcmp(traces.key, name), 'file_id');
    file_id = cell{1,1}{1};
    
    filepath = fullfile('D:\mag_track\processed', ['extracted_vicon_', name, '.csv']);
    
    data = readtable(filepath);
    pos = table2array(data(:,2:4)) / 1000; % NOTE: Assuming mm
    quat = table2array(data(:,5:8));
end