close all
am = 100:-0.01:0;
am = [am,0:-0.01:-100];
time = 0:0.01:200.01;
figure();
plot(time, am.*cos(time))
% hold on;
% plot(time,50*cos(time))

figure();
plot(time, sign(am.*cos(time)))
figure();
plot(time,sign(50*cos(time)))