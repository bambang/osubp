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
% The following structure members are required by the 3DSAR data loader:
% .pass      % 1 through 8
% .pol       % string: HH, VV, HV, VH
% .minaz     % 1 to 360
% .maxaz     % 1 to 360
% .af_flag   % 1 or 0 (only available for HH and VV)
% .dvdPath   % string giving data directory, e.g., 
%            %   '/data/GOTCHA/PublicReleaseDataset3DSAR/DVD', which contains 
%            %   DATA and DOCUMENTATION directories
%
% Ahmed Fasih, The Ohio State University, adapted form LeRoy Gorham, AFRL 

% Define input data parameters here
loader_params = {'pass','pol','minaz','maxaz','af_flag', 'dvdPath'};
for i=1:length(loader_params)
    eval(sprintf('%s = data.%s;', loader_params{i}, loader_params{i}));
end

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

% Setup imaging grid
data.x_vec = linspace(data.x0 - data.Wx/2, data.x0 + data.Wx/2, data.Nx);
data.y_vec = linspace(data.y0 - data.Wy/2, data.y0 + data.Wy/2, data.Ny);
[data.x_mat,data.y_mat] = meshgrid(data.x_vec,data.y_vec);
data.z_mat = zeros(size(data.x_mat));

% Setup space for final image
data.im_final = zeros(size(data.x_mat));

