clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script produces images using the 2D/3D challenge problem        %
% datset.  It uses the imaging function bpBasic for image              %
% formation.  This script assumes a regular imaging grid with z = 0.   %
%                                                                      %
% Written by LeRoy Gorham, Air Force Research Laboratory, WPAFB, OH    %
% Email:  leroy.gorham@wpafb.af.mil                                    %
% Date Released:  8 Apr 2010                                           %
%                                                                      %
% Gorham, L.A. and Moore, L.J., "SAR image formation toolbox for       %
%   MATLAB,"  Algorithms for Synthetic Aperture Radar Imagery XVII     %
%   7669, SPIE (2010).                                                 %
%                                                                      %
% Casteel, C.H., et al, "A challenge problem for 2D/3D imaging of      %
%   targets from a volumetric data set in an urban environment,"       %
%   Altorithms for Synthetic Aperture Radar Imagery XIV 6568, SPIE     %
%   (2007).                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INPUT PARAMETERS START HERE %

% Define the path to the base directory of the dataset
dvdPath = '/ssip1/GOTCHA/PublicReleaseDataset3DSAR/DVD';

% Define input data parameters here
pass = 1;               % What pass to image (1-8)
pol = 'HH';             % What polarization to image (HH,HV,VH,VV)
minaz = 39;             % Minimum azimuth angle (degrees)
maxaz = 42;             % Maximum azimuth angle (degrees)
af_flag = 0;            % Use autofocus flag (Only available for HH and VV)
taper_flag = 0;         % Add a hamming taper for sidelobe control

% Define image parameters here
data.Wx = 100;          % Scene extent x (m)
data.Wy = 100;          % Scene extent y (m)
data.Nfft = 4096;       % Number of samples in FFT
data.Nx = 501;          % Number of samples in x direction
data.Ny = 501;          % Number of samples in y direction
data.x0 = 0;            % Center of image scene in x direction (m)
data.y0 = 0;            % Center of image scene in y direction (m)
dyn_range = 70;         % dB of dynamic range to display

% INPUT PARAMETERS END HERE %

% Determine data path
datadir = sprintf('%s%sdata',dvdPath,filesep);

% Read in the data
for ii = minaz:maxaz
    % Determine file name based on input parameters
    in_fname = sprintf('%s%spass%d%s%s%sdata_3dsar_pass%d_az%03d_%s',datadir,...
        filesep,pass,filesep,pol,filesep,pass,ii,pol);
    
    % Load in the file
    newdata = load(in_fname);

    % If this is the first data file, define new variables to store data.
    % Otherwise, append the data file to the existing variables
    if isfield(data,'phdata')
        % Determine the number of pulses in this data file
        Nin = size(newdata.data.fp,2);
        
        % Determine the number of pulses already added
        Ncur = size(data.phdata,2);

        % Update the phase history
        data.phdata(:,(Ncur+1):(Ncur+Nin)) = newdata.data.fp;
        
        % Update r0, x, y, and z (all in meters)
        data.R0((Ncur+1):(Ncur+Nin)) = newdata.data.r0;
        data.AntX((Ncur+1):(Ncur+Nin)) = newdata.data.x;
        data.AntY((Ncur+1):(Ncur+Nin)) = newdata.data.y;
        data.AntZ((Ncur+1):(Ncur+Nin)) = newdata.data.z;
        
        % Update the autofocus parameters
        data.r_correct((Ncur+1):(Ncur+Nin)) = newdata.data.af.r_correct;
        data.ph_correct((Ncur+1):(Ncur+Nin)) = newdata.data.af.ph_correct;
    else
        % Create new variables for the new data
        data.phdata = newdata.data.fp;
        data.R0 = newdata.data.r0;
        data.AntX = newdata.data.x;
        data.AntY = newdata.data.y;
        data.AntZ = newdata.data.z;
        data.r_correct = newdata.data.af.r_correct;
        data.ph_correct = newdata.data.af.ph_correct;
        data.freq = newdata.data.freq;
    end
end

% Calculate the minimum frequency for each pulse (Hz)
data.minF = min(data.freq)*ones(size(data.R0));

% Calculate the frequency step size (Hz)
data.deltaF = diff(data.freq(1:2));

if af_flag
    % r_correct is a correction applied to r0 (effectivley producing a
    % phase ramp for each pulse
    data.R0 = data.R0 + data.r_correct;
    
    % ph_correct is a phase correction applied to each sample in a pulse
    data.phdata = data.phdata .* repmat(exp(1i*data.ph_correct),[size(data.phdata,1) 1]);
end

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
data = bpBasic(data);

% Display the image
figure
imagesc(data.x_vec,data.y_vec,20*log10(abs(data.im_final)./...
    max(max(abs(data.im_final)))),[-dyn_range 0])
colormap gray
axis xy image;
set(gca,'XTick',-50:25:50,'YTick',-50:25:50);
h = xlabel('x (m)');
set(h,'FontSize',14,'FontWeight','Bold');
h = ylabel('y (m)');
set(h,'FontSize',14,'FontWeight','Bold');
colorbar
set(gca,'FontSize',14,'FontWeight','Bold');
print -deps2 /ssip2/lgorham/SPIE10/fig/3DsarBPA.eps