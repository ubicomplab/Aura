classdef BaseFieldModel
    %BASEFIELDMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = BaseFieldModel()
            %BASEFIELDMODEL Construct an instance of this class
            %   Detailed explanation goes here
        end
        
        function Visualize(obj)
            RANGE = 1;
            INTERVAL = .1;
            BLOCK = .2;

            x = [-RANGE:INTERVAL:-BLOCK, BLOCK:INTERVAL:RANGE];
            y = [-RANGE:INTERVAL:-BLOCK, BLOCK:INTERVAL:RANGE];
            z = -.2:.02:.2;
            [X,Y,Z] = meshgrid(x, y, z);
            X = reshape(X, [], 1);
            Y = reshape(Y, [], 1);
            Z = reshape(Z, [], 1);
            pos = [X, Y, Z];
            field = obj.Evaluate(pos);

            field = nthroot(field, 3);
            field_x = field(:,1);
            field_y = field(:,2);
            field_z = field(:,3);

            figure
            quiver3(X, Y, Z, field_x, field_y, field_z)
            view(-35,45)
        end
        
        function field = GenUniformSample(obj)
            RANGE = 1;
            INTERVAL = .01;

            x = [-RANGE:INTERVAL:RANGE];
            y = [-RANGE:INTERVAL:RANGE];
            z = [-RANGE:INTERVAL:RANGE];
            [X,Y,Z] = meshgrid(x, y, z);
            X = reshape(X, [], 1);
            Y = reshape(Y, [], 1);
            Z = reshape(Z, [], 1);
            pos = [X, Y, Z];
            
            field = obj.Evaluate(pos);
        end
        
    end
    
    methods(Abstract)
        field = Evaluate(position);
    end
end

