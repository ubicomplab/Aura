%% Calculate inductance of multi-layer coil
%  d=wire diameter, u=turns/layer, v=number of layers
%  r0=radius of innermost layer
%  kx=axial pitch, ky=radial pitch
%  All dimensions are in cm, inductance result is in microHenries
function [L] = multicoil(d,v,u,r0,kx,ky)
    g = exp(-0.25) * d/2;
    m = 0;
    nxMin = 1;
  
    % Calculate all mutual inductances
    for ny = (0:1:v-1)
        for nx = (nxMin:1:u-1)
            % multiplication factor
            if (ny == 0 || nx == 0)
                mf = 2;
            else
                mf = 4;
            end
            
            x = nx*kx;
            mult = mf*(u-nx);
            
            for y = (0:1:v-ny-1)
                r1 = r0 + y*ky;
                r2 = r0 + (y+ny)*ky;
                m = m + mult*mut(r1,r2,x);
            end
        end
        nxMin = 0;
    end
    
    % Calculate all self inductances
    for y = (0:1:v-1)
        r1 = r0 + y*ky;
        m = m + u*mut(r1,r1,g);
    end
    
    L = m;
end