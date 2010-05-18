function [data] = generateData3DSAR(data)
% This function takes as input a structure of input parameters for the AFRL
% bpBasic imaging toolkit, loads the raw phase history and flightpath data
% associated with those parameters, and returns these in the structure for
% downstream processing. 
%
% The following structure members are required by the imager itself:
% .Wx
% .Wy
% .Nfft 
% .Nx
% .Ny
% .x0
% .y0
%
% The following structure members are required by the MTI data loader:
% .channel      % 1, 2, 3
% .misrefFlag   % 1 for Mission Image, 0 for Reference Image
% .startPulse   % integer
% .numPulses    % integer
% .dvdPath      % string giving data directory, e.g., 
%             %   '/data/GOTCHA/PublicReleaseDataset3DSAR/DVD', which contains 
%             %   DATA and DOCUMENTATION directories
%
% Ahmed Fasih, The Ohio State University, adapted form LeRoy Gorham, AFRL 

% Define input data parameters here
loader_params = {'channel','misrefFlag','startPulse','numPulses','dvdPath'};
for i=1:length(loader_params)
    eval(sprintf('%s = data.%s;', loader_params{i}, loader_params{i}));
end

prm.startPulse = startPulse;  % Starting pulse
prm.numPulses = numPulses;    % Number of pulses to image

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
data.phdata = readChallengePH(ph_fname,prm);

% Load in Aux Information
load(aux_fname);

% Load PAUX Information
paux = readPAUXData(paux_fname);

% Load GPS Information
data.GPS_struct = load(GPS_fname);

% Setup imaging grid
data.x_vec = linspace(data.x0 - data.Wx/2, data.x0 + data.Wx/2, data.Nx);
data.y_vec = linspace(data.y0 - data.Wy/2, data.y0 + data.Wy/2, data.Ny);
[data.x_mat,data.y_mat] = meshgrid(data.x_vec,data.y_vec);
data.z_mat = zeros(size(data.x_mat));

% Setup space for final image
data.im_final = zeros(size(data.x_mat));

% Set the number of samples per pulse
data.K = size(data.phdata,1);

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

