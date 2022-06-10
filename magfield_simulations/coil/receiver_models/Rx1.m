classdef Rx1 < BaseRxModel
    %RX1 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = Rx1()
            %RX1 Construct an instance of this class
            %   Detailed explanation goes here
        end
        
        function coilPos = GenCoilPositions(obj, pos, rot)
            coilPos(:,:, 1) = pos;
            coilPos(:,:, 2) = zeros(size(pos,1), 3);
            coilPos(:,:, 3) = zeros(size(pos,1), 3);
        end
    end
end

