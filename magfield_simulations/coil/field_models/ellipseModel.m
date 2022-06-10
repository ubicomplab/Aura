classdef ellipseModel < BaseFieldModel
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Position
        Rotation
    end
    
    methods
        function obj = ellipseModel(position,rotation)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.Position = position;
            obj.Rotation = rotation;
        end
        
        function field = Evaluate(obj,position,radius)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            pos_world = position - obj.Position;
            pos_coil = inv(obj.Rotation) * pos_world';
            
            BSmag = BSmag_init();
            % parametrise transmitter coil: circular filament centred at origin
            D_STEP = 1000;  % discretization increment [rad]

            t = linspace(-pi, pi, D_STEP);
            a = radius(1); % horizontal radius
            b = radius(2); % vertical radius
            x0 = 0;%tx_pos(1);
            y0 = 0;%tx_pos(2);
            L = zeros(length(t),3);
            for i = 1:length(t)
                L(i,:) = [x0 + a*cos(t(i)), y0 + b*sin(t(i)), 0];
            end
            figure(); plot(L(:,1),L(:,2));
            % add the transmitter coil; let I = 1 for M calculations
            BSmag = BSmag_add_filament(BSmag, L, 1, min(radius)*2*pi/D_STEP);


            % plot receiver sample locations
            % BSmag_plot_field_points(BSmag, pos_coil(1,:), pos_coil(2,:), pos_coil(3,:));

            % calculate B at specified receiver location
            [BSmag, X,Y,Z, BX, BY, BZ] = BSmag_get_B(BSmag, pos_coil(1,:), pos_coil(2,:), pos_coil(3,:));
            field = (obj.Rotation *[BX; BY; BZ] / 1.4259e-11)';
        end
    end
end

