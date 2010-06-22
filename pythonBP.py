# Ohio State PyCUDA SAR Backprojection demo script
# Extending and utilizing AFRL bpBasic Matlab package
#
# Ahmed Fasih, fasih.1@osu.edu
# see accompanying Mercurial repo for history

from mlabwrap import mlab
import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule
from numpy import float64
FloatType = numpy.float32

clight = 299792458.0
# Params
block_width = 16
block_height = 16

# Useful functions
mdouble = mlab.double;
a2f = lambda x: float(mdouble(x)[0,0])
nint    = numpy.int32
nfloat  = numpy.float32

def grabsource(fi):
    """Read an ascii file and return its contents.
    
    Note: no checks are performed!"""
    f = open(fi, 'r')
    src = f.read()
    f.close()
    return src
def complex_to_2chan(x):
    """Convert a complex array into a CUDA 2-channel array. 
    
    PyCUDA should support this soon (if it doesn't already)."""
    return numpy.array(numpy.dstack((x.real, x.imag)), 
            dtype=numpy.float32, order='C', copy=True) 
def four_to_4chan(x,y,z,w):
    """Converts 4 1D arrays into a 4-channel CUDA array.
    
    Used for packing aircraft location information into a texture."""
    return numpy.array(numpy.dstack((x,y,z,w)), dtype=numpy.float32, order='C',copy=True)
def show_image(im):
    """Matlab-specific function to plot a dB image nicely."""
    mlab.figure()
    mlab.imagesc(data.x_vec, data.y_vec, 20*numpy.log10(numpy.abs(im)))
    mlab.colormap('green14',nout=0)
    mlab.axis('image', nout=0)
    mlab.colorbar(nout=0)
    mlab.axis('xy',nout=0)
    mlab.caxis(numpy.max(mlab.caxis()) - numpy.array([a2f(data.dyn_range), 0]), nout=0)


# Use Matlab to load and pre-process the data. Returns "data" struct
data = mlab.helper3DSAR() # AFRL Volumetric Dataset
#data = mlab.helperMTI()   # AFRL GMTI Dataset

# Perform FFT/upsampling/windowing in Matlab
data = mlab.rangeCompress(data)


# Unpack "data" structure from Matlab
rp = mdouble(data.upsampled_range_profiles)

im = numpy.zeros_like(data.im_final).astype(numpy.complex64)
[Nimg_height, Nimg_width] = im.shape
delta_pixel_x = numpy.diff(data.x_vec)[0,0]
delta_pixel_y = numpy.diff(data.y_vec)[0,0]
pi_4_f0__clight = (numpy.pi * 4.0 * (mdouble(data.minF) / clight)).astype(numpy.float32)

R0 = float64(clight) / (2.0 * float64(a2f(data.deltaF)))
Nfft = nint(a2f(data.Nfft))
rmin = -R0 / 2.0
rmax = (Nfft * 0.5 - 1.0) * R0 / float(Nfft)

platform_info = four_to_4chan(mdouble(data.AntX), mdouble(data.AntY), 
        mdouble(data.AntZ), mdouble(data.R0))

# Load CUDA source file
src = grabsource('PyCUDABackProjectionKernel.cu')
mod = SourceModule(src, 
        include_dirs=['.'])

# Set up CUDA texture for range projections
tex_projections = mod.get_texref('tex_projections')
arr_projections = drv.make_multichannel_2d_array(complex_to_2chan(rp),
        order='C')
tex_projections.set_filter_mode(drv.filter_mode.LINEAR)
tex_projections.set_array(arr_projections)

# Run!
backprojection_loop = mod.get_function('backprojection_loop')
backprojection_loop(
        drv.Out(im), 
        nint(a2f(data.Np)),
        nint(Nimg_height),
        nfloat(delta_pixel_x), 
        nfloat(delta_pixel_y),
        nint(a2f(data.Nfft)),
        drv.In(pi_4_f0__clight),
        nfloat(numpy.min(data.x_vec)),
        nfloat(numpy.min(data.y_vec)),
        drv.In(platform_info),
        nfloat(rmin), 
        nfloat(rmax),
        block=(block_width,block_height,1),
        grid=(Nimg_width/block_width, Nimg_height/block_height),
        texrefs=[tex_projections]);


show_image(im)
mlab.title('PyCUDA BP')

if 1:
    adata = mlab.bpBasic(data)
    show_image(adata.im_final)
    mlab.title('bpBasic, single')
    show_image(adata.im_final - im)
    mlab.title('bpBasic - CUDA')



