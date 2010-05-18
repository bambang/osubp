function [data] = helperMTI()

data.clight = 299792458.0;

% For imager and loader
data.Wx = 200;
data.Wy = 200;
data.Nfft = 1024*4;
data.Nx = 128*4;
data.Ny = 128*4;
data.x0 = -25;
data.y0 = -50;

% For loader
data.channel = 1;
data.misrefFlag = 1;
data.startPulse = 1;
data.numPulses = 1000;
data.dvdPath = '/home/aldebrn/gradschool/GMTI_public_dataset';

% Extra stuff
data.taper_flag = 0;
data.dyn_range = 50;

data = generateDataMTI(data);

end
