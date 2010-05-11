clear all; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script generates synthetic point targets and calls the          %
% imaging function bpBasic for image formation.  This script assumes   %
% a regular imaging grid with z = 0.  It also assumes a circular       %
% flight path with constant R0.                                        %
%                                                                      %
% Written by LeRoy Gorham, Air Force Research Laboratory, WPAFB, OH    %
% Email:  leroy.gorham@wpafb.af.mil                                    %
% Date Released:  8 Apr 2010                                           %
%                                                                      %
% Gorham, L.A. and Moore, L.J., "SAR image formation toolbox for       %
%   MATLAB,"  Algorithms for Synthetic Aperture Radar Imagery XVII     %
%   7669, SPIE (2010).                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% INPUT PARAMETERS START HERE %

% Define sensor parameters here
sensor.BW = 600e6;           % Bandwidth (Hz)
sensor.Fc = 10*1e9;          % Center freq (Hz)
sensor.int_angle = 3;        % Integration Angle (degrees)
sensor.cent_angle = 50;      % Center Angle (degrees)
sensor.elev = 30;            % Elevation Angle (degrees)
sensor.R0 = 10e3;            % Range to sensor (m)
data.K = 512;                % Number of frequency samples
data.Np = 128;               % Number of pulses

% Define image parameters here
data.Wx = 10;                % Scene extent x (m)
data.Wy = 10;                % Scene extent y (m)
data.Nfft = 4096;            % Number of samples in FFT
data.Nx = 501;               % Number of samples in x direction
data.Ny = 501;               % Number of samples in y direction
dyn_range = 50;              % dB of dynamic range to display

% Define the point targets here
targPos = [0,0,0;
          -3,2,0;
           1,4,0];           % Point target positions (m)
amp = [1 1 1];               % Point target amplitudes (~voltage)

% INPUT PARAMETERS END HERE %

% Define speed of light (m/s)
c = 299792458;

% Calculate the frequency vector (Hz)
data.freq = linspace(sensor.Fc-sensor.BW/2,sensor.Fc+sensor.BW/2,data.K)';

% Calculate the azimuth angle of the sensor at each pulse
sensor.azim = linspace(sensor.cent_angle-sensor.int_angle/2,sensor.cent_angle+sensor.int_angle/2,data.Np); % (Degrees)

% Calculate the x,y,z position of the sensor at each pulse
[data.AntX,data.AntY,data.AntZ] = sph2cart(sensor.azim*pi/180,...
    ones(1,data.Np)*sensor.elev*pi/180,ones(1,data.Np)*sensor.R0);

% Determine the number of point targets
N_targets = size(targPos,1);

% Loop through each pulse to calculate the phase history
for ii = 1:length(sensor.azim)
    
    % Initialize the vector which contains the phase history for this pulse
    freqdata = zeros(data.K,1);
    
    % Loop through each target
    for kk = 1:N_targets
        % Calculate the differential range to the target (m)
        dR = sqrt((data.AntX(ii)-targPos(kk,1))^2+...
            (data.AntY(ii)-targPos(kk,2))^2+...
            (data.AntZ(ii)-targPos(kk,3))^2) - sensor.R0;
        
        % Update the phase history for this pulse
        freqdata = freqdata + amp(kk) * exp(-1i*4*pi*dR/c*data.freq);
    end
    
    % Put the phase history into the data structure
    data.phdata(:,ii) = freqdata;
end

% Define the imaging grid using MATLAB function meshgrid
data.x_vec = linspace(-data.Wx/2,data.Wx/2,data.Nx);
data.y_vec = linspace(-data.Wy/2,data.Wy/2,data.Ny);
[data.x_mat,data.y_mat] = meshgrid(data.x_vec,data.y_vec);
data.z_mat = zeros(size(data.x_mat));

% Calculate R0 for each pulse (m)
data.R0 = sensor.R0 * ones(size(data.AntX));

% Calculate the frequency step size (Hz)
data.deltaF = diff(data.freq(1:2));

% Calculate the minimum frequency (Hz)
data.minF = data.freq(1) * ones(size(data.AntX));

% Call the backprojection function with the appropriate inputs
data = bpBasic(data);

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
print(gcf,'-deps2', '/ssip2/lgorham/SPIE10/fig/ptTargBPA.eps');