classdef Coil
    %COIL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        L
        C
        R
        R_extra
        diameter
        n_turns_total
    end
    
    methods
        function obj = Coil(L, C, R, R_extra, diameter, n_turns_total)
            obj.L = L;
            obj.C = C;
            obj.R = R;
            obj.R_extra = R_extra;
            obj.diameter = diameter;
            obj.n_turns_total = n_turns_total;
        end
        
        function a = area(obj)
            a = pi * (obj.diameter/2)^2;
        end
    end
    
    methods(Static)
        function c = FromWindings(diameter, num_turns_per_layer, num_layers, gauge, F_hz, R_extra, C)
            [L, R, ~] = Lcalc(gauge, diameter*1000, num_turns_per_layer, num_layers, F_hz*1e-3);
            if ~exist('C','var')
                C = 1/(2*pi*F_hz)^2 / L;
            end
            c = Coil(L, C, R, R_extra, diameter, num_turns_per_layer*num_layers);
        end
        function c = FromL(diameter, num_turns_per_layer, num_layers, gauge, F_hz, L, R_extra, C)
            [~, R, ~] = Lcalc(gauge, diameter*1000, num_turns_per_layer, num_layers, F_hz*1e-3);
            if ~exist('C','var')
                C = 1/(2*pi*F_hz)^2 / L;
            end
            c = Coil(L, C, R, R_extra, diameter, num_turns_per_layer*num_layers);
        end
    end
            
    
end

