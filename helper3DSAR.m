function [data] = helper3DSAR()

data.clight = 299792458.0;

% For imager and loader
data.Wx = 100;
data.Wy = 100;
data.Nfft = 1024*4;
data.Nx = 128*1;
data.Ny = 128*1;
data.x0 = 0;
data.y0 = 0;

% For loader
data.pass = 1;
data.pol = 'HH';
data.minaz = 1;
data.maxaz = 3;
data.af_flag = 1;
data.dvdPath = '/home/aldebrn/gradschool/Volumetric_SAR_public_dataset';

% Extra stuff
data.taper_flag = 0;
data.dyn_range = 50;

data = generateData3DSAR(data);

end
