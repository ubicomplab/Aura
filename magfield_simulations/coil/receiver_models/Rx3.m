classdef Rx3 < BaseRxModel
    %RX1 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = Rx3()
            %RX1 Construct an instance of this class
            %   Detailed explanation goes here
        end
        
        function coilPos = GenCoilPositions(obj, pos, rot)
            c1 = [0,0,0];
            c2 = [0, 0, 0.05];
            c3 = [.04, 0.02, .06];
            coilPos = zeros(size(pos, 1), 3, 3);
            coilPos(:,:, 1) = pos + quatrotate(rot, c1);
            coilPos(:,:, 2) = pos + quatrotate(rot, c2);
            coilPos(:,:, 3) = pos + quatrotate(rot, c3);
        end
    end
end

