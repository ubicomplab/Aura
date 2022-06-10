TX_D = 8e-2;
RX_LOC = [0,0,12] * 1e-2;
B = Bfield(TX_D, 1, RX_LOC(1), RX_LOC(2), RX_LOC(3)); % [T]

gs = GlobalSearch();

%       AWG,   D,         NT,     NL,    C,            f,     BX,BY,BZ
x0 =   [26,26, TX_D,2e-2, 20,20,  6,3,  117e-9,3.2e-6,10e3, B(1),B(2),B(3)];
% x0 =   [26,26, TX_D,4e-2, 25,10,  5,5,   5e-9,27e-9,   10e3,  B(1),B(2),B(3)];
xmin = [20,24, TX_D,2e-2, 10,5,   1,1,   1e-12,1e-12,  1e3,   B(1),B(2),B(3)];
xmax = [38,38, TX_D,3e-2, 100,50, 20,10, 1,1,          100e3, B(1),B(2),B(3)];
opts = optimoptions(@fmincon, 'UseParallel', true);

problem = createOptimProblem('fmincon', 'x0', x0, 'objective', @measure, ...
    'lb', xmin, 'ub', xmax, 'options', opts);

[x, fval] = run(gs, problem);

TX_AWG = x(1);
RX_AWG = x(2);
TX_D = x(3);
RX_D = x(4);
TX_NT = x(5);
RX_NT = x(6);
TX_NL = x(7);
RX_NL = x(8);
TX_C = x(9);
RX_C = x(10);
f = x(11);
B = [x(12), x(13), x(14)];
