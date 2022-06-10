classdef BaseRxModel
    %BASERXMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = BaseRxModel()
            %BASERXMODEL Construct an instance of this class
            %   Detailed explanation goes here
        end

        function [measurements] = Measure(obj, fieldModel, pos, rot,radius)
            coilPos = obj.GenCoilPositions(pos, rot);
            measurements = zeros(size(pos,1), 3, size(coilPos, 3));
            for rxCoil = 1:size(coilPos,3)
                fields = fieldModel.Evaluate(coilPos(:,:,rxCoil),radius);
                dirX = quatrotate(rot, [1,0,0]);
                dirY = quatrotate(rot, [0,1,0]);
                dirZ = quatrotate(rot, [0,0,1]);
                if size(rot, 1) == 1
                    measurements(:, 1, rxCoil) = (fields * dirX');
                    measurements(:, 2, rxCoil) = (fields * dirY');
                    measurements(:, 3, rxCoil) = (fields * dirZ');
                else
                    measurements(:, 1, rxCoil) = dot(fields, dirX, 2);
                    measurements(:, 2, rxCoil) = dot(fields, dirY, 2);
                    measurements(:, 3, rxCoil) = dot(fields, dirZ, 2);
                end
            end
            measurements = (measurements);
            
        end
    end
    methods(Static)
        function [pos, rot] = GenTransforms(N, space_count)
           pos = BaseRxModel.GenPositions(N, space_count);
           disp(size(pos,1));
           rot = BaseRxModel.GenRotations(N, size(pos,1));
        end
        function quaternions = GenRotations(N, posSize)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            % https://en.wikipedia.org/wiki/Rotation_matrix#Uniform_random_rotation_matrices
            quaternions = randn(posSize, 4);
            % quaternions = repmat([1,0,0, 0],N, 1);
            quaternions = quaternions ./ vecnorm(quaternions, 2, 2);
            
            % quaternions = angle2quat(rand(N,1)*2*pi, zeros(N, 1), zeros(N, 1));
        end
        function [pos, logic] = GenPositions(N, space_count)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here


%             unit_pos = (rand(N, 3) - .5) * 2;
%             unit_pos = unit_pos ./ vecnorm(unit_pos, 2, 2);
%             
%             minX = (minR / maxR).^3;
%             maxX = 1;
%             x = ((rand(N, 1) * (maxX - minX)) + minX);
%             rad = nthroot(x, 3) * maxR;
%             
%             pos = rad .* unit_pos;
            
                
% Eighth
%             x1 = ((rand([N,1])) * 1) - .5;
%             y1 = ((rand([N,1])) * .6) - .7;
%             z1 = ((rand([N,1])) * .5) + .95;
% Half
%             x1 = ((rand([N,1])) * 1) - .5;
%             y1 = ((rand([N,1])-.5) * 2 * .6) - .4;
%             z1 = ((rand([N,1])-.5) * 2 * .5) + .7;
% Full
            pos = [];
            while(size(pos, 1) < N)
                x = ((rand([N, 1])-.5) * 2 * .8);
                y = ((rand([N, 1])- 1) * .8);
                z = ((rand([N, 1])-1)) * .8;
                pos_temp = [x,y,z];
    %             if X_POS == -1
    %                 logic = x(:,1)<0 & distance > (space_count)*MAX_DIST/SPACE_DIV & distance < (space_count+1)*MAX_DIST/SPACE_DIV;
    %             elseif X_POS == 1
    %                 logic = x(:,1)>0 & distance > (space_count)*MAX_DIST/SPACE_DIV & distance < (space_count+1)*MAX_DIST/SPACE_DIV;
    %             end
                distance = vecnorm(pos_temp,2,2);
%                 switch space_count
%                     case 1
%                         logic = z(:,1) > 0 & distance > 0.125 & distance < 0.25; %000
%     %                     logic = distance>0.125 & distance< 0.25; %000
%                     case 2
%                         logic = z(:,1) > 0 & distance > 0.25 & distance < 0.5; %001
%     %                     logic = distance>0.25 & distance< 0.5; %001
%                     case 3
%                         logic = z(:,1) > 0 & distance > 0.5;
%     %                     logic = distance>0.5 & distance< 0.75; %001
%                     case 4
%                         logic = z(:,1) < 0 & distance > 0.125 & distance < 0.25; %000
%                     case 5
%                         logic = z(:,1) < 0 & distance > 0.25 & distance < 0.5; %001
%                     case 6
%                         logic = z(:,1) < 0 & distance > 0.5; %001
%                 end
%                 pos = [pos; x(logic,1), y(logic,1), z(logic,1)];
%                 disp(size(pos));
                pos = [pos; x(:,1), y(:,1), z(:,1)];
            end
        end
    end
    
    methods(Abstract)
        coilPos = GenCoilPositions(obj, pos, rot);
    end
end

