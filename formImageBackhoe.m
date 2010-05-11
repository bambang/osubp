clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script produces images using the Visual-D Backhoe dataset.  It  %
% uses the imaging function bpBasicFarField for image formation.  This %
% script assumes a regular imaging grid with z = 0.                    %
%                                                                      %
% Each file of the dataset contains multiple elevation slices.  There  %
% is a parameter called elevSamp which determines which slice to use   %
% within a file.  Selecting 8 gives the "middle" elevation (i.e., if   %
% you want to image 30 degrees elevation, the 8th sample in the file   %
% corresponds to exactly 30 degrees.                                   %
%                                                                      %
% Written by LeRoy Gorham, Air Force Research Laboratory, WPAFB, OH    %
% Email:  leroy.gorham@wpafb.af.mil                                    %
% Date Released:  8 Apr 2010                                           %
%                                                                      %
% Gorham, L.A. and Moore, L.J., "SAR image formation toolbox for       %
%   MATLAB,"  Algorithms for Synthetic Aperture Radar Imagery XVII     %
%   7669, SPIE (2010).                                                 %
%                                                                      %
% Naidu, K. and Lin, L., "Data dome:  full k-space sampling data for   %
%   high-frequency radar research,"  Algorithms for Synthetic Aperture %
%   Radar Imagery XI 5427, SPIE (2004).                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INPUT PARAMETERS START HERE %

% Define the path to the base directory of the dataset
basePath = '/ssip2/visuald_data/backhoe';

% Define input data parameters here
elev = 10;              % What elevation to image (deg)
elevSamp = 8;           % Which elevation sample in the file to image 
                        %     (8 gives the middle elevation)
pol = 'VV';             % What polarization to image (HH,VHHV,VV)
minaz = 1;              % Minimum azimuth angle (degrees)
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
datadir = sprintf('%s%s%ddeg%smat_outputs',basePath,filesep,elev,filesep);

% Read in the data
for ii = minaz:maxaz
    % Determine file name based on input parameters
    in_fname = sprintf('%s%sbackhoe_el%03d_az%03d.mat',datadir,...
        filesep,elev,ii);
    
    % Load in the file
    newdata = load(in_fname);
    
    % If this is the first data file, define new variables to store data.
    % Otherwise, append the data file to the existing variables
    if isfield(data,'phdata')
        % Determine the number of pulses in this data file
        Nin = size(newdata.data.vv,2);
        
        % Determine the number of pulses already added
        Ncur = size(data.phdata,2);

        % Update the phase history
        switch pol
            case{'HH'}
                data.phdata(:,(Ncur+1):(Ncur+Nin)) = squeeze(newdata.data.hh(:,elevSamp,:));
            case{'VV'}
                data.phdata(:,(Ncur+1):(Ncur+Nin)) = squeeze(newdata.data.vv(:,elevSamp,:));
            case{'VHHV'}
                data.phdata(:,(Ncur+1):(Ncur+Nin)) = squeeze(newdata.data.vhhv(:,elevSamp,:));
        end
        
        % Update Antenna Azimuth and Elevation Angles (degrees))
        data.AntAzim((Ncur+1):(Ncur+Nin)) = newdata.data.azim(elevSamp,:);
        data.AntElev((Ncur+1):(Ncur+Nin)) = newdata.data.elev(elevSamp,:);
    else
        % Create new variables for the new data
        switch pol
            case{'HH'}
                data.phdata = squeeze(newdata.data.hh(:,elevSamp,:));
            case{'VV'}
                data.phdata = squeeze(newdata.data.vv(:,elevSamp,:));
            case{'VHHV'}
                data.phdata = squeeze(newdata.data.vhhv(:,elevSamp,:));
        end
        data.AntAzim = newdata.data.azim(elevSamp,:);
        data.AntElev = newdata.data.elev(elevSamp,:);
        data.freq = newdata.data.FGHz * 1e9;
    end
end

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
print -deps2 /ssip2/lgorham/SPIE10/fig/backhoeBPA.eps