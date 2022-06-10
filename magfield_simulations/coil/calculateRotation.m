c2 = [0,0,0.5];
dif =(coilPos2(:,:)-coilPos1(:,:));
for i = 1:size(dif,1)
    a = dot(dif(i,:),dif(i,:));
    normDot(i,:) = sqrt(dot(c2,c2)*a);
    real_part = normDot(i,:) + dot(c2, dif(i,:));
    w(i,:) = -1 * cross(c2,dif(i,:));
    qq(i,:) = [real_part,w(i,:)];
    q_normal(i,:) = qq(i,:) ./ vecnorm(qq(i,:),2,2);
    
    rotx(i,1) = atan2( dif(i,2), dif(i,3) );
    if (dif(i,3) >= 0) 
        roty(i,1) = -1*atan2( dif(i,1) * cos(rotx(i,1)), dif(i,3));
    else
        roty(i,1) = atan2( dif(i,1) * cos(rotx(i,1)), -dif(i,3));
    end
    rotz(i,1) = atan2(cos(rotx(i,1)), sin(rotx(i,1)) * sin(roty(i,1)) );
end

eul_est = [rotz(:,1),roty(:,1),rotx(:,1)];
eul_from_quat_normal = quat2eul(q_normal(:,:));
eul = quat2eul(rot(:,:));

pitch = [];
yaw = [];
roll = [];
for i=1:size(eul_est,1)
    if ((abs(eul_est(i,1)) - abs(eul(i,1))) < 10e-5)
        roll = [roll, i];
    end
    if ((abs(eul_est(i,2)) - abs(eul(i,2))) < 10e-5)
        pitch = [pitch,i];
    end
    if ((abs(eul_est(i,3)) - abs(eul(i,3))) < 10e-5)
        yaw = [yaw,i];
    end 
end

