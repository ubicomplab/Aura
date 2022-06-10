for i=1:size(rot,1)  
    sqx(i) = rot(i,2)*rot(i,2);
    sqy(i) = rot(i,3)*rot(i,3);
    sqz(i) = rot(i,4)*rot(i,4);
    yaw_actual(i) = atan2(2*(rot(i,1)*rot(i,2)+rot(i,3)*rot(i,4)) , 1 - 2*sqx(i) - 2*sqy(i));
    pitch_actual(i) = asin(2*(rot(i,1)*rot(i,3)-rot(i,2)*rot(i,4)));
    roll_actual(i) = atan2(2*(rot(i,1)*rot(i,4)+rot(i,2)*rot(i,3)) , 1 - 2*sqy(i) - 2*sqz(i));
end