function pos = get_AEPosition(p, m, n, ma, na, TRx, Coordinate)
% =========================================================================
% -- Function to compute the positions of AE in m-th row and n-th column of
% SA in Tx/Rx
% =========================================================================

% -- Function: p = get_AEPositions(p, m, n, TRx)
% -- Input Arguments:
%       p: Channel struct that contains the channel parameters
%       m: Index of receiver/transmitter SAs (rows) based on TRx
%       n: Index of receiver/transmitter SAs (columns) based on TRx
%       ma: Index of receiver/transmitter AEs (rows) based on TRx
%       na: Index of receiver/transmitter AEs (columns) based on TRx
%       TRx: Defines the direction of the link, 'T' at Tx side, 'R' at Rx side
%       Coordinate: Type of coordinate, 'L' is local, 'G' is global

% -- Output Arguments:
%       a: Global positions of AEs, a 1D-Array of size(3,1)

%=================================================

% -- (c) 2021 Simon Tarboush, Hadi Sarieddeen, Hui Chen, 
%             Mohamed Habib Loukil, Hakim Jemaa, 
%             Mohamed-Slim Alouini, Tareq Y. Al-Naffouri

% -- e-mail: simon.w.tarboush@gmail.com; hadi.sarieddeen@kaust.edu.sa; hui.chen@kaust.edu.sa;
%            mohamedhabib.loukil@kaust.edu.sa; hakim.jemaa@kaust.edu.sa;
%            slim.alouini@kaust.edu.sa; tareq.alnaffouri@kaust.edu.sa

% =========================================================================

% S. Tarboush, H. Sarieddeen, H. Chen, M.-H. Loukil, H. Jemaa, M.-S. Alouini, and T. Y. Al-Naffouri, 
%  "TeraMIMO:  A  channel  simulator for  wideband  ultra-massive  MIMO  terahertz  communications," 
%  arXivpreprint arXiv:2104.11054, 2021.

% =========================================================================
%% Initialize Output
if strcmp(TRx, 'R')
    
    positionLocalSA = [0; (n-1-(p.Nr-1)/2)*p.DeltaNr; (m-1-(p.Mr-1)/2)*p.DeltaMr];
    positionLocalAE = positionLocalSA + [0; (na-1-(p.Nar-1)/2)*p.deltaNr; (ma-1-(p.Mar-1)/2)*p.deltaMr];
    positionGlobal = p.RotmRx*positionLocalAE + p.positionRx;
    
elseif strcmp(TRx, 'T')
    
    positionLocalSA = [0; (n-1-(p.Nt-1)/2)*p.DeltaNt; (m-1-(p.Mt-1)/2)*p.DeltaMt];
    positionLocalAE = positionLocalSA + [0; (na-1-(p.Nat-1)/2)*p.deltaNt; (ma-1-(p.Mat-1)/2)*p.deltaMt];
    positionGlobal = p.RotmTx*positionLocalAE + p.positionTx;
else
    
    error('TRx has only two options: T/R');
end

if strcmp(Coordinate, 'G')
    pos = positionGlobal;
elseif strcmp(Coordinate, 'L')
    pos = positionLocalAE;
else
    error('Coordinate has only two options: L/G');
end
