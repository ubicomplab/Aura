function [pos, quat] = gen_trace(N)
    FILENAME = 'pos_gen';
    header = 'x,y,z,qw,qx,qy,qz';
    pos = rand(1,3)*0.5+0.25;
    for i = 1:1:N
        pos_index = size(pos,1);
        temp = rand(1,3)*0.5+0.25;
        m = temp- pos(pos_index,:);
        dist = round(abs(vecnorm(m*1000,2,2)));
        for j = 0:1:dist
            pos = [pos; pos(pos_index,:) + (j/dist).*m]; 
        end
        if mod(i,500)==0
            disp(i)
        end
    end
    figure();
    scatter3(pos(:,1),pos(:,2),pos(:,3))
    hold on;
    quat = zeros(size(pos,1),4);
    
    file = ['D:\mag_track\sim\', FILENAME, '.csv'];

    fid = fopen(file,'w'); 
    fprintf(fid,'%s\n',header);
    fclose(fid);
    dlmwrite(file, [pos,quat], '-append');
end