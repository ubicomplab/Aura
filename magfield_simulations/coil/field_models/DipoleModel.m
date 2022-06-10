classdef DipoleModel < BaseFieldModel
    %DIPOLEMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Position
        Rotation
    end
    
    methods
        function obj = DipoleModel(position, rotation)
            %DIPOLEMODEL Construct an instance of this class
            %   Detailed explanation goes here
            obj.Position = position;
            obj.Rotation = rotation;
        end
        
        function field = Evaluate(obj, position, radius)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            pos_world = position - obj.Position;
            pos_coil = inv(obj.Rotation) * pos_world';

            % x = r*sin(theta);
            % y = r*sin(theta)*cos(phi);
            % z = r*cos(theta);
            x = pos_coil(1,:);
            y = pos_coil(2,:);
            z = pos_coil(3,:);

            r = sqrt(x.^2 + y.^2 + z.^2);
            Bx = 3*x.*z./(r.^5);
            By = 3*y.*z./(r.^5);
            Bz = (3*z.^2-r.^2)./(r.^5);
            field = (obj.Rotation * [Bx;By;Bz])';

        end
    end
end

