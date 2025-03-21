% =========================================================================
% -- Script to generate a time-invariant (TIV) THz channel
% =========================================================================

% -- (c) 2021 Simon Tarboush, Hadi Sarieddeen, Hui Chen, 
%             Mohamed Habib Loukil, Hakim Jemaa, 
%             Mohamed-Slim Alouini, Tareq Y. Al-Naffouri

% -- e-mail: simon.w.tarboush@gmail.com; hadi.sarieddeen@kaust.edu.sa; hui.chen@kaust.edu.sa;
%            mohamedhabib.loukil@kaust.edu.sa; hakim.jemaa@kaust.edu.sa;
%            slim.alouini@kaust.edu.sa; tareq.alnaffouri@kaust.edu.sa

% =========================================================================

% S. Tarboush, H. Sarieddeen, H. Chen, M.-H. Loukil, H. Jemaa, M.-S. Alouini, and T. Y. Al-Naffouri, 
%  "TeraMIMO: A channel simulator for wideband ultra-massive MIMO terahertz communications,"
%  arXivpreprint arXiv:2104.11054, 2021.

% =========================================================================

%%
clear; clc;
close all;


%% Add Paths
% path(pathdef); addpath(pwd);
% cd Channel; addpath(genpath(pwd)); cd ..;
% cd Molecular_Absorption;addpath(genpath(pwd)); cd ..; 
% cd Visualization;addpath(genpath(pwd)); cd ..; 


%% Initialize Parameters

% Transmission Parameters
p.channelType = 'Multipath+LoS';  % Options: /'LoS' /'Multipath' /'Multipath+LoS'
p.addrandomness = 'Off';    % 'On', default is 'Off'

p.Fc = 0.3e12;          % Center frequency of the transmission bandwidth (Hz)
p.BW = 0.01e12;         % Total channel bandwidth (Hz)
p.Nsub_c = 2^0;         % Number of subcarriers to divide the total Bandwidth (K-subcarriers)  
p.Nsub_b = 2^0;         % Number of sub-bands in each subcarrier

% SAs
p.Mt = 2;               % Number of transmitter SAs (row)
p.Nt = 2;               % Number of transmitter SAs (column)
p.Mr = 2;               % Number of Receiver SAs (row)
p.Nr = 2;               % Number of Receiver SAs (column)

p.DeltaMt = 1e-2;       % Spacing between rows of SAs @Tx
p.DeltaNt = 1e-2;       % Spacing between columns of SAs @Tx
p.DeltaMr = 1e-2;       % Spacing between rows of SAs @Rx
p.DeltaNr = 1e-2;       % Spacing between columns of SAs @Rx

% AEs
p.Mat = 2;              % Number of transmitter AEs (row) inside each SA
p.Nat = 2;              % Number of transmitter AEs (column) inside each SA
p.Mar = 2;              % Number of receiver AEs (row) inside each SA
p.Nar = 2;              % Number of receiver AEs (column) inside each SA

p.deltaMt = 5e-4;       % Spacing between rows of AEs @Tx
p.deltaNt = 5e-4;       % Spacing between columns of AEs @Tx
p.deltaMr = 5e-4;       % Spacing between rows of AEs @Rx
p.deltaNr = 5e-4;       % Spacing between columns of AEs @Rx


% Spherical and planar wave model
% Supported combinations for this version are (SA/AE): Plane/Plane, Sphere/Plane, Sphere/Sphere (NO steering vector)
p.WaveModelSA = 'Sphere';        %'Sphere'/'Plane'
p.WaveModelAE = 'Sphere';        %'Sphere'/'Plane'


% Geometry design
% Define local/global position and Euler angles
p.positionTx = [0; 0; 0];       % Tx center 3D positions (global coordinates)
p.eulerTx    = [0; 0; 0];       % Tx Euler rotation angles, following ZYX intrinsic rotation
p.positionRx = [1; 0; 0];       % Rx center 3D positions (global coordinates)
p.eulerRx    = [pi; 0; 0];      % Rx Euler rotation angles, following ZYX intrinsic rotation


% Update channel parameters
p_ch = update_channel_param_TIV(p);


%% Calculation of Absorption Coefficient 
K_abs = compute_Abs_Coef(p_ch);


%% Call Channel
% H = channel_TIV_AE_freq_domain(p_ch, K_abs); 
d_ray = get_RayleighDistance(p_ch);
d_fres = get_FresnelDistance(p_ch);
% p.positionRx = [d_ray; 0; 0];
% p_ch = update_channel_param_TIV(p);
% plot_GeometryAE(p_ch)
H = channel_TIV_AE_freq_domain_spherical(p_ch, K_abs); 
norm(H,'fro')

% % Number of subcarriers
% num_subcarriers = size(H,3);
% 
% % Calculate the number of rows and columns for the subplots based on the number of subcarriers
% nRows = ceil(sqrt(num_subcarriers));
% nCols = ceil(num_subcarriers / nRows);
% 
% % Create a single figure
% figure;
% 
% for i = 1:num_subcarriers
%     % Create a subplot for each subcarrier
%     subplot(nRows, nCols, i);
%     imagesc(abs(H(:,:,i)));
%     colorbar; % Optional: display a colorbar for reference
%     xlabel('Tx AE index')
%     ylabel('Rx AE index')
%     title(['subcarrier ',num2str(i)])
% end

% Number of subcarriers
num_subcarriers = size(H,3);

% Calculate the number of rows and columns for the subplots based on the number of subcarriers
nRows = ceil(sqrt(num_subcarriers));
nCols = ceil(num_subcarriers / nRows);

% Determine the global minimum and maximum values for abs(H)
global_min = min(abs(H(:)));
global_max = max(abs(H(:)));

% Create a single figure
figure;

for i = 1:num_subcarriers
    % Create a subplot for each subcarrier
    subplot(nRows, nCols, i);
    imagesc(abs(H(:,:,i)), [global_min, global_max]); % set consistent color limits
    xlabel('Tx AE index')
    ylabel('Rx AE index')
    title(['subcarrier ',num2str(i)])
end

% Add a single colorbar for the entire figure
h=colorbar;
set(h, 'Position', [.93 .11 .0181 .8150])
for i=1:nCols*nRows
    pos=get(subplot(nRows,nCols,i),'Position');
    pos(3)=0.93*pos(3);
    set(subplot(nRows,nCols,i), 'Position', pos);
end

% norm(H(:,:,1),'fro')
% H_p = H;
% H_s = H;
% for i = 1:size(H,3)
%     figure;
%     imagesc(abs(H(:,:,i)));
%     colorbar; % Optional: display a colorbar for reference
%     xlabel('Tx AE index')
%     ylabel('Rx AE index')
%     title(['subcarrier ',num2str(i)])
% end
% H = CH_Response.H;
% size_H = size(H)

% h = CH_Response.h;
% size_h = size(h)
% 
% H111 = H(1,1,1)
% h111 = h(1,1,1)
%% Visualize Channel
% Plot_TIV_THz_Channel(p_ch, CH_Response.H, CH_Response.h);

