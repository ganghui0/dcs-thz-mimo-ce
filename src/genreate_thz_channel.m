function genreate_thz_channel(n_ch, d, delta, Delta, type, max_angle)
% =========================================================================
% -- Produce NMSE in dB for different estimation techniques and save to
% "NMSE_SNR_xx_xx_Approach.mat"
% =========================================================================
tic
% clear
% clc
% close all
% addpath(genpath('./src/TeraMIMO-main'));
% path
% pc = parcluster('local');
% parpool(pc, str2num(getenv('SLURM_CPUS_ON_NODE')));
%% Initialize Parameters
% Transmission Parameters
% p.channelType = 'Multipath+LoS';  % Options: /'LoS' /'Multipath' /'Multipath+LoS'
p.channelType = type;

p.Fc = 0.3e12;          % Center frequency of the transmission bandwidth (Hz)
p.BW = 0.01e12;         % Total channel bandwidth (Hz)
p.Nsub_c = 2^3;         % Number of subcarriers to divide the total Bandwidth (K-subcarriers)  
p.Nsub_b = 2^0;         % Number of sub-bands in each subcarrier

% SAs
p.Mt = 2;               % Number of transmitter SAs (row)
p.Nt = 2;               % Number of transmitter SAs (column)
p.Mr = 2;               % Number of Receiver SAs (row)
p.Nr = 2;               % Number of Receiver SAs (column)

% p.DeltaMt = 1e-2;       % Spacing between rows of SAs @Tx
% p.DeltaNt = 1e-2;       % Spacing between columns of SAs @Tx
% p.DeltaMr = 1e-2;       % Spacing between rows of SAs @Rx
% p.DeltaNr = 1e-2;       % Spacing between columns of SAs @Rx

p.DeltaMt = Delta;       % Spacing between rows of SAs @Tx
p.DeltaNt = Delta;       % Spacing between columns of SAs @Tx
p.DeltaMr = Delta;       % Spacing between rows of SAs @Rx
p.DeltaNr = Delta;       % Spacing between columns of SAs @Rx
% AEs
p.Mat = 8;              % Number of transmitter AEs (row) inside each SA
p.Nat = 8;              % Number of transmitter AEs (column) inside each SA
p.Mar = 4;              % Number of receiver AEs (row) inside each SA
p.Nar = 2;              % Number of receiver AEs (column) inside each SA

% p.deltaMt = 5e-4;       % Spacing between rows of AEs @Tx
% p.deltaNt = 5e-4;       % Spacing between columns of AEs @Tx
% p.deltaMr = 5e-4;       % Spacing between rows of AEs @Rx
% p.deltaNr = 5e-4;       % Spacing between columns of AEs @Rx
p.deltaMt = delta;       % Spacing between rows of AEs @Tx
p.deltaNt = delta;       % Spacing between columns of AEs @Tx
p.deltaMr = delta;       % Spacing between rows of AEs @Rx
p.deltaNr = delta;       % Spacing between columns of AEs @Rx


% Spherical and planar wave model
% Supported combinations for this version are (SA/AE): Plane/Plane, Sphere/Plane, Sphere/Sphere (NO steering vector)
p.WaveModelSA = 'Sphere';        %'Sphere'/'Plane'
p.WaveModelAE = 'Sphere';        %'Sphere'/'Plane'


% Geometry design
% Define local/global position and Euler angles
% d = 1.2;
p.positionTx = [0; 0; 0];       % Tx center 3D positions (global coordinates)
p.eulerTx    = [0; 0; 0];       % Tx Euler rotation angles, following ZYX intrinsic rotation
p.positionRx = [d; 0; 0];       % Rx center 3D positions (global coordinates)
p.eulerRx    = [pi; 0; 0];      % Rx Euler rotation angles, following ZYX intrinsic rotation


% Update channel parameters
p = update_channel_param_TIV(p);

% Calculate the direction vector between Tx and Rx
% direction = (p.positionRx - p.positionTx) / norm(p.positionRx - p.positionTx);
%% Calculation of Absorption Coefficient 
K_abs = compute_Abs_Coef(p);

%% Monte Carlo simulation
% n_ch = 1000;

% Power = 1; % power
K = p.Nsub_c; % Number of subcarriers
N_t = p.Qt * p.Qat; % Number of Tx dAEs;
N_r = p.Qr * p.Qar; % Number of Rx AEs;

% max_angle = pi/6;

H = zeros(N_r,N_t,K,n_ch); % true channel
sprintf("n_ch: %d, d: %g, delta: %g, Delta: %g, type: %s, angle: %g",n_ch, d, p.deltaMt, p.DeltaMt, p.channelType, max_angle)
parfor n = 1:n_ch
    % n
    p_local = p;
    angle_shift = -max_angle + (max_angle - (-max_angle)) * rand(3,1);
    p_local.eulerRx = p_local.eulerRx + angle_shift;
    p_local = update_channel_param_TIV(p_local);

    % Compute channel realization
    H_local = channel_TIV_AE_freq_domain_spherical(p_local, K_abs);
    H(:,:,:,n) = H_local;
end
filename = sprintf('data/channel-r%dt%dk%d-n%dd%gdelta%gDelta%gtheta%g-%s.mat', N_r, N_t, p.Nsub_c, n_ch, d, delta, Delta, max_angle, p.channelType);
save(filename, 'H','-v7.3'); % use v7.3 to save variable larger than 2GB
toc
end