clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script produces images using the Civilian Vehicle Radar Data    %
% Domes dataset.  It uses the imaging function bpBasicFarField for     %
% image formation.  This script assumes a regular imaging grid with    %
% z = 0.                                                               %
%                                                                      %
% Written by LeRoy Gorham, Air Force Research Laboratory, WPAFB, OH    %
% Email:  leroy.gorham@wpafb.af.mil                                    %
% Date Released:  8 Apr 2010                                           %
%                                                                      %
% Gorham, L.A. and Moore, L.J., "SAR image formation toolbox for       %
%   MATLAB,"  Algorithms for Synthetic Aperture Radar Imagery XVII     %
%   7669, SPIE (2010).                                                 %
%                                                                      %
% Dungan, K.E., et al, "Civilian vehicle radar data domes," Algorithms %
%   for Synthetic Aperture Radar Imagery XVII 7669, SPIE (2010).       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INPUT PARAMETERS START HERE %

% Define the path to the base directory of the dataset
basePath = '/ssip1/GOTCHA/CVdomes';

% Define input data parameters here
target = 'Jeep93';      % What target to image
elev = 30;              % What elevation to image (deg)
pol = 'VV';             % What polarization to image (HH,HV,VV)
minaz = 0;              % Minimum azimuth angle (degrees)
maxaz = 360;            % Maximum azimuth angle (degrees)
taper_flag = 0;         % Add a hamming taper for sidelobe control

% Define image parameters here
data.Wx = 10;           % Scene extent x (m)
data.Wy = 10;           % Scene extent y (m)
data.Nfft = 8192;       % Number of samples in FFT
data.Nx = 501;          % Number of samples in x direction
data.Ny = 501;          % Number of samples in y direction
data.x0 = 0;            % Center of image scene in x direction (m)
data.y0 = 0;            % Center of image scene in y direction (m)
dyn_range = 70;         % dB of dynamic range to display

% INPUT PARAMETERS END HERE %

% Determine data path
datadir = sprintf('%s%sDomes%s%s',basePath,filesep,filesep,target);

% Determine filename
fname = sprintf('%s%s%s_el%6.4f.mat',datadir,filesep,target,elev);

% Load data
newdata = load(fname);

% Determine which pulses are located within specified azimuth angles
I = find(and(newdata.data.azim >= minaz, newdata.data.azim <= maxaz));

% Update the phase history
switch pol
    case{'HH'}
        data.phdata = newdata.data.hh(:,I);
    case{'VV'}
        data.phdata = newdata.data.vv(:,I);
    case{'HV'}
        data.phdata = newdata.data.hv(:,I);
end

% Update other parameters needed for imaging
data.AntAzim = newdata.data.azim(I);
data.AntElev = newdata.data.elev * ones(size(data.AntAzim));
data.freq = newdata.data.FGHz * 1e9;

% Calculate the minimum frequency for each pulse (Hz)
data.minF = min(data.freq)*ones(size(data.AntAzim));

% Calculate the frequency step size (Hz)
data.deltaF = diff(data.freq(1:2));

% Determine the number of pulses and the samples per pulse
[data.K,data.Np] = size(data.phdata);

% Add a hamming taper to the data if desired
if taper_flag
    data.phdata = data.phdata .* (hamming(data.K)*hamming(data.Np)');
end

% Setup imaging grid
data.x_vec = linspace(data.x0 - data.Wx/2, data.x0 + data.Wx/2, data.Nx);
data.y_vec = linspace(data.y0 - data.Wy/2, data.y0 + data.Wy/2, data.Ny);
[data.x_mat,data.y_mat] = meshgrid(data.x_vec,data.y_vec);
data.z_mat = zeros(size(data.x_mat));

% Call the backprojection function with the appropriate inputs
data = bpBasicFarField(data);

% Display the image
figure
imagesc(data.x_vec,data.y_vec,20*log10(abs(data.im_final)./...
    max(max(abs(data.im_final)))),[-dyn_range 0])
colormap gray
axis xy image;
set(gca,'XTick',-5:5,'YTick',-5:5);
h = xlabel('x (m)');
set(h,'FontSize',14,'FontWeight','Bold');
h = ylabel('y (m)');
set(h,'FontSize',14,'FontWeight','Bold');
colorbar
set(gca,'FontSize',14,'FontWeight','Bold');
print -deps2 /ssip2/lgorham/SPIE10/fig/CVdomesBPA.eps