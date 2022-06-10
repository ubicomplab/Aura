%% calculate inductance of a coil
%  awg = wire gauge, dci = coil diameter [mm]
%  nt = number of turns/layer, nl = number of layers
%  f = frequency of AC magnetic field [kHz]
function [L, R, lca] = Lcalc(awg, dci, nt, nl, f)
    % dci = dci*1e3;
    % disp('warning, dci changed to mm');
    f = f*1e-3;

    dw = 0.127 * 92^((36-awg)/39);  % conductor diameter [mm]
    aw = pi/4 * dw^2;               % conductor cross-section area [mm^2]
    di = odCalc(dw);                % wire diameter [mm]
    
    if (nt>0 && nl>0 && dci>0 && dw>0)
        muo = 4*pi*1e-7;
        rs = 1.72e-8; % resistivity of copper [Wm]
        
        % axial length calculations (coil length)
        sp = di / (pi*dci);
        psic = atan(sp / sqrt(1-sp^2));
        pa = tan(psic) * pi*dci;
        lca = pa*nt; % [mm]
        
        % radial pitch calculations calculatiosn (coil depth)
        pr = di;
        lcr = pr*nl;
        dco = dci + 2*nl*pr;
        
        % polygonal parameters (circle)
        pre = pr;
        rei = dci/2;
        
        Ls = multicoil(dw,nl,nt,rei,pa,pre)*1e3;    % uH
        Lcor = 0;           % Roundwire corrections not required

        % geometry parameters
        circi = pi*dci;     % inner circumference of winding [mm]
        circo = pi*dco;     % outer circumference of winding [mm]
        circ = (circi + circo) / 2;     % mean circumference [mm]
        psi = atan(pa/circ);            % mean pitch angle
        lw = nt * nl * circ/cos(psi);   % total length of wire [mm]
        
        % Frequency correction due to skin effect
        if (f > 0)
            sk = sqrt(rs / (pi^2*0.0000004*f)); % skin depth at operating frequency [m]
            s = sk/(dw*1e-3);                   % ratio of skin depth to wire diameter
            Li = LiCorr(s)*1e6 * lw*1e-3;       % uH
        else
            Li=0;
       end
        
        % final calculations
        L = (Ls + Lcor + Li) * 1e-6; % H
        R = rs*1e3 * lw / (aw);   % Ohms
    end
end