function [pos, quat] = load_gen_trace(name)

    filepath = fullfile('D:\mag_track\sim', [name, '.csv']);
    data = readtable(filepath);
    pos = table2array(data(:,1:3));
    quat = table2array(data(:,4:7));
end