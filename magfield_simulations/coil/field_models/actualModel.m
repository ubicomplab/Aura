classdef actualModel < BaseFieldModel
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Position
        Rotation
    end
    
    methods
        function obj = actualModel(position,rotation)
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.Position = position;
            obj.Rotation = rotation;
        end
        
        function field = Evaluate(obj,position,radius)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            BSmag = BSmag_init();

            % parametrise transmitter coil: circular filament centred at origin
            D_STEP = 1000;  % discretization increment [rad]

            figure(); scatter3(obj.Position(:,1),obj.Position(:,2),obj.Position(:,3));


            % add the transmitter coil; let I = 1 for M calculations
            BSmag = BSmag_add_filament(BSmag, obj.Position, 1, 0.01/D_STEP);

            % plot receiver sample locations
            BSmag_plot_field_points(BSmag, position(:,1), position(:,2), position(:,3));

            % calculate B at specified receiver location
            [BSmag, X,Y,Z, BX, BY, BZ] = BSmag_get_B(BSmag, position(:,1), position(:,2), position(:,3));
            field = [BX BY BZ];
        end
    end
end

