clear all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script produces images using the SAR-based GMTI urban dataset.  %
% It uses the imaging function bpBasic for image formation.  This      %
% script assumes a regular imaging grid with z = 0.                    %
%                                                                      %
% Written by LeRoy Gorham, Air Force Research Laboratory, WPAFB, OH    %
% Email:  leroy.gorham@wpafb.af.mil                                    %
% Date Released:  8 Apr 2010                                           %
%                                                                      %
% Gorham, L.A. and Moore, L.J., "SAR image formation toolbox for       %
%   MATLAB,"  Algorithms for Synthetic Aperture Radar Imagery XVII     %
%   7669, SPIE (2010).                                                 %
%                                                                      %
% Scarborough, S.M., et al, "A challenge problem for SAR-based GMTI in %
%   urban environments,"  Algorithms for Synthetic Aperture Radar      %
%   Imagery, XVI 7337, SPIE (2009).                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INPUT PARAMETERS START HERE %

% Define the path to the base directory of the dataset
dvdPath = '/ssip1/GOTCHA/PublicReleaseGMTI';

% Define input data parameters here
channel = 1;                  % Which channel to image (1-3)
misrefFlag = 1;               % 1 for Mission Image, 0 for Reference Image
taper_flag = 0;               % Add a hamming taper for sidelobe control
prm.startPulse = 1;           % Starting pulse
prm.numPulses = 8000;         % Number of pulses to image

% Define image parameters here
data.Wx = 200;                % Scene extent x (m)
data.Wy = 200;                % Scene extent y (m)
data.Nfft = 4096;             % Number of samples in FFT
data.Nx = 1001;               % Number of samples in x direction
data.Ny = 1001;               % Number of samples in y direction
data.x0 = -25;                % Center of image scene in x direction (m)
data.y0 = -50;                % Center of image scene in y direction (m)
dyn_range = 70;               % dB of dynamic range to display

% INPUT PARAMETERS END HERE %

% Data specific parameters
Korig = 5400;                 % The number of valid samples in the original data

% Set path to the code to read in data
codePath = sprintf('%s%sMatlab Codes',dvdPath,filesep);
addpath(codePath);

% Set path to the data
datadir = sprintf('%s%sSPIEchallengeData',dvdPath,filesep);

% Define speed of light (m/s)
c = 299792458;

% Setup File Names
if misrefFlag
    ph_fname = sprintf('%s%sdurangoChallenge_chan%d_mis2_PH',datadir,filesep,channel);
    aux_fname = sprintf('%s_auxSaveData.mat',ph_fname);
    paux_fname = sprintf('%s%sdurangoChallenge_mis2_PAUX',datadir,filesep);
else
    ph_fname = sprintf('%s%sdurangoChallenge_chan%d_ref4_PH',datadir,filesep,channel);
    aux_fname = sprintf('%s_auxSaveData.mat',ph_fname);
    paux_fname = sprintf('%s%sdurangoChallenge_ref4_PAUX',datadir,filesep);
end
GPS_fname = sprintf('%s%sdurangoChallenge_GPStruth.mat',datadir,filesep);

% Load in Phase History
PH = readChallengePH(ph_fname,prm);

% Load in Aux Information
load(aux_fname);

% Load PAUX Information
paux = readPAUXData(paux_fname);

% Load GPS Information
load(GPS_fname);

% Setup imaging grid
data.x_vec = linspace(data.x0 - data.Wx/2, data.x0 + data.Wx/2, data.Nx);
data.y_vec = linspace(data.y0 - data.Wy/2, data.y0 + data.Wy/2, data.Ny);
[data.x_mat,data.y_mat] = meshgrid(data.x_vec,data.y_vec);
data.z_mat = zeros(size(data.x_mat));

% Add a hamming taper to the data if desired
if taper_flag
    data.phdata = PH .* (hamming(data.K)*hamming(data.Np)');
else
    data.phdata = PH;
end

% Set the number of samples per pulse
data.K = size(PH,1);

% Set the number of pulses
data.Np = prm.numPulses;

% Determine the index to the pulses that are going to be imaged
I = prm.startPulse:(prm.startPulse+data.Np-1);

% Set the (x,y,z) values of the antenna for the given pulses (m)
data.AntX = paux.APCpos(1,I);
data.AntY = paux.APCpos(2,I);
data.AntZ = paux.APCpos(3,I);

% Set the range to scene center for each pulse (m)
data.R0 = paux.APCMag(I);

% Calculate the frequency step size for the ORIGINAL data (Hz)
deltaF = paux.Gamma(1)/aux_SPIE.readPrm.Fsamp;

% Calculate the frequency step size in this dataset (Hz)
data.deltaF = deltaF * Korig / data.K;

% Calculate the minimum frequency for each pulse in the dataset (Hz)
startFreqSamp = -aux_SPIE.circshiftFloorPHradius(I) + ...
    aux_SPIE.PHradius_sampleError + 1;
data.minF = paux.PHradius(I) + startFreqSamp * deltaF;

% Calculate the step size in the range profile (m)
deltaR = c/(2*Korig*deltaF);

% Calculate the new R0 based on the range bin offset (m)
data.R0 = data.R0 + aux_SPIE.readPrm.deltaRbin(I) * deltaR;

% Call the backprojection function with the appropriate inputs
data = bpBasic(data);

% Display the image
figure
imagesc(data.x_vec,data.y_vec,20*log10(abs(data.im_final)./...
    max(max(abs(data.im_final)))),[-dyn_range 0])
colormap gray
axis xy image;
h = xlabel('x (m)');
set(h,'FontSize',14,'FontWeight','Bold');
h = ylabel('y (m)');
set(h,'FontSize',14,'FontWeight','Bold');
colorbar
set(gca,'FontSize',14,'FontWeight','Bold');
print -deps2 /ssip2/lgorham/SPIE10/fig/mtiBPA.eps