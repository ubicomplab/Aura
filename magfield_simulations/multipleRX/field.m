function [B] = field(P)

% x = r*sin(theta);
% y = r*sin(theta)*cos(phi);
% z = r*cos(theta);
x = P(1);
y = P(2);
z = P(3);

r = sqrt(x.^2 + y.^2 + z.^2);
Bx = 3*x.*z/(r.^5);
By = 3*y.*z/(r.^5);
Bz = (3*z.^2-r.^2)/(r.^5);
B = [Bx,By,Bz];
end

