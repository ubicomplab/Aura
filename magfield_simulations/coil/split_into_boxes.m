function data = split_into_boxes(pos, features)
    distance = vecnorm(pos, 2, 2);
    indices = 1:length(pos);
    logic = [];
%     logic = [logic, distance >= 200 & distance < 400 & pos(:,1) > 0];
%     logic = [logic, distance >= 400 & distance < 500 & pos(:,1) > 0];
%     logic = [logic, distance >= 500 & distance < 600 & pos(:,1) > 0];
%     logic = [logic, distance >= 600 & distance < 700 & pos(:,1) > 0];
%     logic = [logic, distance >= 700 & distance < 800 & pos(:,1) > 0];
    logic = [logic, distance >= 200 & distance < 500 & pos(:,1) > 0];
    logic = [logic, distance >= 500 & distance < 800 & pos(:,1) > 0];
    logic = [logic, distance >= 200 & distance < 500 & pos(:,1) < 0];
    logic = [logic, distance >= 500 & distance < 800 & pos(:,1) < 0];
%     logic = [logic, distance >= 200 & distance < 800 & pos(:,1) > 0];
%     logic = [logic, distance >= 200 & distance < 800 & pos(:,1) < 0];
    function s = package(box_logic)
       s = {};
       s.pos = pos(box_logic==1, :);
       s.features = features(box_logic==1, :);
       s.indices = indices(box_logic==1);
    end
    
    data = {};
    for logic_idx = 1:size(logic, 2)
        data{end+1} = package(logic(:,logic_idx));
    end

end