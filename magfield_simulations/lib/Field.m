classdef Field
    %FIELD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        BSmag
        B
        L
    end
    
    properties (Constant)
      D_STEP = 100
    end
    
    methods
        function obj = Field(position, tx_coil, R, P)
            obj.BSmag = BSmag_init();

            % parametrise transmitter coil: circular filament centred at origin
            a = linspace(0,2*pi, Field.D_STEP);
            obj.L = zeros(length(a),3);
            for i = 1:length(a)
                obj.L(i,:) = R * ((tx_coil.diameter/2)*[cos(a(i)), sin(a(i)), 0])' + P';
            end

            % add the transmitter coil; let I = 1 for M calculations
            % for i = 1:TX_N
            %     BSmag = BSmag_add_filament(BSmag, L, 1, D_STEP);
            % end
            obj.BSmag = BSmag_add_filament(obj.BSmag, obj.L, 1, 1/Field.D_STEP);

            % plot receiver sample locations
            % BSmag_plot_field_points(obj.BSmag, position(1), position(2), position(3));

            % calculate B at specified receiver locations
            [obj.BSmag, ~,~,~, X, Y, Z] = BSmag_get_B(obj.BSmag, position(:,1), position(:,2), position(:,3));
            obj.B = [X, Y, Z] * tx_coil.n_turns_total;     % T, assuming current = 1 A, so really it's T/A

        end
    end
    
end

