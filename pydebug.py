from mlabwrap import mlab
import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule

mlab.addpath('../FasihSarStuff/', nout=0)

# Params
clight = 299792458.0
block_width = 16
block_height = 16

# Matlab data loading
data = mlab.helper3DSAR()
#data = mlab.helperMTI()

data = mlab.rangeCompress(data)

mdouble = mlab.double;
do = lambda x: float(mdouble(x)[0,0])
nint    = numpy.int32
nfloat  = numpy.float32

rp = mdouble(data.upsampled_range_profiles)

im = numpy.zeros_like(data.im_final).astype(numpy.complex64)
[Nimg_height, Nimg_width] = im.shape
delta_pixel_x = numpy.diff(data.x_vec)[0,0]
delta_pixel_y = numpy.diff(data.y_vec)[0,0]
c__4_deltaF = clight / (4.0 * do(data.deltaF))
R_start_pre = c__4_deltaF  * do(data.Nfft) / (do(data.Nfft)-1.0)
pi_4_f0__clight = (numpy.pi * 4.0 * (mdouble(data.minF) / clight)).astype(numpy.float32)



# CUDA sources
def grabsource(fi):
    f = open(fi, 'r')
    src = f.read()
    f.close()
    return src

src = grabsource('PyCUDABackProjectionKernel.cu')
mod = SourceModule(src, 
        include_dirs=['.','/home/aldebrn/NVIDIA_GPU_Computing_SDK/C/common/inc',
            '/home/aldebrn/matlab2009a/extern/include'])

def complex_to_2chan(x):
    return numpy.array(numpy.dstack((x.real, x.imag)), 
            dtype=numpy.float32, order='C', copy=True) 
def four_to_4chan(x,y,z,w):
    return numpy.array(numpy.dstack((x,y,z,w)), dtype=numpy.float32, order='C',copy=True)

tex_projections = mod.get_texref('tex_projections')
arr_projections = drv.make_multichannel_2d_array(complex_to_2chan(rp), order='C') 
tex_projections.set_filter_mode(drv.filter_mode.LINEAR)
tex_projections.set_array(arr_projections)


tex_platform_info = mod.get_texref('tex_platform_info')
arr_platform_info = drv.make_multichannel_2d_array(four_to_4chan(
    mdouble(data.AntX), mdouble(data.AntY), mdouble(data.AntZ), 
    mdouble(data.R0)), order='C')
tex_platform_info.set_array(arr_platform_info)

platform_info = four_to_4chan(mdouble(data.AntX), mdouble(data.AntY), 
        mdouble(data.AntZ), mdouble(data.R0))

# height, width, num_channels for order == 'C'

# Testing
if 0:
    xyzr = lambda id: numpy.array([data.AntX[0,id], data.AntY[0,id], data.AntZ[0,id], data.R0[0,id]])

    testing_platform = mod.get_function('testing_platform')
    ansx = numpy.array([[0]]).astype(numpy.float32)
    ansy = numpy.array([[0]]).astype(numpy.float32)
    ansz = numpy.array([[0]]).astype(numpy.float32)
    answ = numpy.array([[0]]).astype(numpy.float32)
    id = 0
    testing_platform(drv.In(platform_info), drv.Out(ansx), drv.Out(ansy), drv.Out(ansz), drv.Out(answ),
            nint(id), block=(1,1,1), texrefs=[tex_platform_info])
    ans = numpy.hstack([ansx,ansy,ansz,answ])
    (numpy.abs(ans -  xyzr(id)))


    testing_platform_tex = mod.get_function('testing_platform_tex')
    ansx = numpy.array([[0]]).astype(numpy.float32)
    ansy = numpy.array([[0]]).astype(numpy.float32)
    ansz = numpy.array([[0]]).astype(numpy.float32)
    answ = numpy.array([[0]]).astype(numpy.float32)
    id = 0
    testing_platform_tex(drv.Out(ansx), drv.Out(ansy), drv.Out(ansz), drv.Out(answ),
            nfloat(id+0.5), block=(1,1,1), texrefs=[tex_platform_info])
    ans = numpy.hstack([ansx,ansy,ansz,answ])
    (numpy.abs(ans -  xyzr(id)))

    rint = lambda x: int(round(x))
    testing_proj_tex = mod.get_function('testing_proj_tex')
    ansre = numpy.array([[0]]).astype(numpy.float32)
    ansim = numpy.array([[0]]).astype(numpy.float32)
    xx = 0
    yy = 5
    def doproj(xx,yy): # pulse num, range bin
        testing_proj_tex(drv.Out(ansre), drv.Out(ansim), 
                nfloat(xx+0.5), nfloat(yy+0.5), 
                block=(1,1,1), texrefs=[tex_projections])
        try:
            return [rp[rint(yy),rint(xx)], ansre[0,0]+1j*ansim[0,0]]
        except:
            return ansre[0,0]+1j*ansim[0,0]
    doproj(0,1002)

# Serious!
from numpy import float64
x_mat = numpy.zeros_like(data.im_final).astype(numpy.float32)
y_mat = numpy.zeros_like(data.im_final).astype(numpy.float32)
debug_effective_idx = numpy.zeros_like(data.im_final).astype(numpy.float32)
debug_dR = numpy.zeros_like(data.im_final).astype(numpy.float32)
R0 = float64(clight) / (2.0 * float64(do(data.deltaF)))
Nfft = nint(do(data.Nfft))
rmin = -R0 / 2.0
rmax = (Nfft * 0.5 - 1.0) * R0 / float(Nfft)

rtest = mod.get_function('testing_r')
ans=numpy.array([0.0], dtype=numpy.float32)
ndo = lambda x: nfloat(do(x))
rtest(nfloat(50.),nfloat(50.), 
        ndo(data.AntX[0]),ndo(data.AntY[0]),ndo(data.AntZ[0]),
        drv.Out(ans), block=(1,1,1), grid=(1,1))
print '%.32g' % ans

backprojection_loop = mod.get_function('backprojection_loop')
backprojection_loop(
        drv.Out(im), 
        nint(do(data.Np)),
        nint(Nimg_height),
        nfloat(delta_pixel_x), 
        nfloat(delta_pixel_y),
        nfloat(R_start_pre),
        nint(do(data.Nfft)),
        nint(0), 
        nint(0),
        nfloat(c__4_deltaF),
        drv.In(pi_4_f0__clight),
        nfloat(numpy.min(data.x_vec)),
        nfloat(numpy.min(data.y_vec)),
        nfloat(0), 
        drv.In(platform_info),
        drv.Out(debug_effective_idx),
        drv.Out(debug_dR),
        drv.Out(x_mat),
        drv.Out(y_mat),
        nfloat(rmin), 
        nfloat(rmax),
        block=(block_width,block_height,1),
        grid=(Nimg_width/block_width, Nimg_height/block_height),
        texrefs=[tex_platform_info, tex_projections]);

def show_image(im):
    mlab.figure()
    mlab.imagesc(data.x_vec, data.y_vec, 20*numpy.log10(numpy.abs(im)))
    mlab.colormap('green14',nout=0)
    mlab.axis('image', nout=0)
    mlab.colorbar(nout=0)
    mlab.axis('xy',nout=0)
    mlab.caxis(numpy.max(mlab.caxis()) - numpy.array([do(data.dyn_range), 0]), nout=0)

show_image(im)
mlab.title('PyCUDA BP')

if 1:
    adata = mlab.bpBasic(data)
    show_image(adata.im_final)
    mlab.title('bpBasic, single')
    show_image(adata.im_final - im)
    mlab.title('bpBasic - CUDA')



