/* vim: set tabstop=4, shiftwidth=4, expandtab */
#include "mex.h"    /* Matlab junk */

#include <stdio.h>  /* printf */
#include "cuda.h"   /* CUDA */
#include <cutil_inline.h>
#include <cutil.h>
#include <time.h>

//#include "PyCUDABackProjectionKernel.h" /* #defines and prototypes and such */



/***
 * Compiler logics
 * **/
#  define MY_CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        printf( "Cuda error in file '%s' in line %i : %s.\n",                \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    } }

#  define MY_CUDA_SAFE_CALL( call)     MY_CUDA_SAFE_CALL_NO_SYNC(call);             \



#define CLIGHT 299792458.0f /* c: speed of light, m/s */
#define PI 3.14159265359f   /* pi, accurate to 6th place in single precision */
#define PI2 6.2831853071800f   /* 2*pi */
#define PI_4__CLIGHT (4.0f * PI / CLIGHT)

#define REAL(vec) (vec.x)
#define IMAG(vec) (vec.y)

#define CAREFUL_AMINUSB_SQ(x,y) __fmul_rn(__fadd_rn((x), -1.0f*(y)), __fadd_rn((x), -1.0f*(y)))

#define ASSUME_Z_0    1        /* Ignore consult_DEM() and assume height = 0. */
#define USE_FAST_MATH 0     /* Use __math() functions? */
#define USE_RSQRT     0

#define MEXDEBUG      1

#define FLOAT_CLASS   mxSINGLE_CLASS

#ifndef VERBOSE
#define VERBOSE       0
#endif

#define BLOCKWIDTH    16
#define BLOCKHEIGHT   16

#define ZEROCOPY      0

#define MAKERADIUS(xpixel,ypixel, xa,ya,za) sqrtf(CAREFUL_AMINUSB_SQ(xpixel, xa) + CAREFUL_AMINUSB_SQ(ypixel, ya) + __fmul_rn(za, za))

/* Pound defines from my PyCUDA implementation:
 * 
 * ---Physical constants---
 * CLIGHT
 * PI
 *
 * ---Radar/data-specific constants---
 * Delta-frequency
 * Number of projections
 *
 * ---Application runtime constants---
 * Nfft, projection length
 * Image dimensions in pixels
 * Top-left image corner
 * X/Y pixel spacing
 *
 * ---Complicated constants---
 * PI_4_F0__CLIGHT = 4*pi/clight * radar_start_frequency
 * C__4_DELTA_FREQ = clight / (4 * radar_delta_frequency)
 * R_START_PRE = C__4_DELTA_FREQ * Nfft / (Nfft-1)
 *
 * ---CUDA constants---
 * Block dimensions
 */



/***
 * Type defs
 * ***/
typedef float FloatType; /* FIXME: this should be used everywhere */

/* From ATK imager */
typedef struct {
    float * real;
    float * imag;
} complex_split;

/* To work seamlessly with Hartley's codebase */
typedef complex_split bp_complex_split;


/***
 * Prototypes
 * ***/

float2 * format_complex_to_columns(bp_complex_split a, int width_orig, 
        int height_orig);

float2 * format_complex(bp_complex_split a, int size);

float4 * format_x_y_z_r(float * x, float * y, float * z, float * r, int size);

void run_bp(bp_complex_split phd, 
        float * xObs, float * yObs, float * zObs, float * r,
        int my_num_phi, int my_proj_length, int nxout, int nyout, 
        int image_chunk_width, int image_chunk_height, 
        int device, 
        bp_complex_split host_output_image,
        int start_output_index, int num_output_rows,
        float c__4_delta_freq, float pi_4_f0__clight, 
        float * start_frequencies,
        float left, float right, float bottom, float top,
        float min_eff_idx, float total_proj_length);

void convert_f0(float * vec, int N);
float extract_f0(float * vec, int N);

__global__ void backprojection_loop(float2 * full_image, 
        int Nphi, int nyout,
        float delta_pixel_x, float delta_pixel_y, float R_START_PRE, 
        int PROJ_LENGTH, int X_OFFSET, int Y_OFFSET,
        float C__4_DELTA_FREQ, float * PI_4_F0__CLIGHT, float left, float bottom,
        float min_eff_idx, float4 * platform_info,
        float * debug_effective_idx, float * debug_2, float * x_mat, float * y_mat,
        float rmin, float rmax);

__device__ float2 expjf(float in);
__device__ float2 expjf_div_2(float in);



/*
void testing_backprojection_loop(float2 * full_image,
        int Nphi, int nyout, float delta_pixel_x, float delta_pixel_y, 
        float R_START_PRE, int PROJ_LENGTH,
        int X_OFFSET, int Y_OFFSET,
        float C__4_DELTA_FREQ, float PI_4_F0__CLIGHT, 
        float left, float bottom, 
        int blockIdxx, int blockIdxy, int threadidxx, int threadidxy, 
        float * output_idxs);
        */




/* Globals and externs */

/* Complex textures containing range profiles */
texture<float2, 2, cudaReadModeElementType> tex_projections;   

/* 4-elem. textures for x, y, z, r0 */
texture<float4, 1, cudaReadModeElementType> tex_platform_info; 

void convert_f0(float * vec, int N) {
    int i;
    for (i=0; i<N; ++i)
        vec[i] *= PI_4__CLIGHT;
}

float extract_f0(float * vec, int N) {
    /* Mean ...
    int i;
    float sum = 0;
    for (i=0; i<N; ++i) {
        sum += vec[i];
    }
    return sum / N;
    */
    return vec[0];
}


/* 
 * Application parameters:
 *  - range profiles
 *
 * 
 * ATK imager gets the following:
 * - range profiles (complex)
 * - f0, vector of start frequencies, Hz
 * - r0, vector of distances from radar to center of illuminated scene, m
 * - x, y, z, vectors of radar position (x points east, y north, z up), m
 * - Nimgx, Nimgy, number of pixels in x and y
 * - deltaf, spacing of frequency vector, Hz
 * - Left, right, top, bottom, corners of the square on the ground to image
 */
void mexFunction(int nlhs,     /* number of LHS (output) arguments */
        mxArray *plhs[],       /* array of mxArray pointers to outputs */
        int nrhs,              /* number of RHS (input) args */
        const mxArray *prhs[]) /* array of pointers to inputs*/
{
    /* Section 1. 
     * These are the variables we'll use */
    /* Subsection A: these come from Matlab and are the same as the ATK code */
    complex_split range_profiles;
    float * start_frequencies;
    float * aimpoint_ranges;
    float * xobs, * yobs, * zobs;
    int Nx_imgwidth, Ny_imgheight;
    float delta_frequency;
    float left, right, top, bottom;

    float min_eff_idx, total_proj_length;

    /* Subsection B: these are computed from the matlab inputs */
    int Npulses, Nrangebins;
    float c__4_delta_freq;
    float pi_4_f0__clight;

    /* Subsection C: these are CUDA-specific options */
    int device, blockwidth, blockheight;

    /* Subsection D: these are output variables */
    complex_split host_output_image;
    
    /* Section 2. 
     * Parse Matlab's inputs */
    range_profiles.real = (float*)mxGetPr(prhs[0]);
    range_profiles.imag = (float*)mxGetPi(prhs[0]); 

    start_frequencies = (float*)mxGetPr(prhs[1]);
    aimpoint_ranges   = (float*)mxGetPr(prhs[2]);
    xobs              = (float*)mxGetPr(prhs[3]);
    yobs              = (float*)mxGetPr(prhs[4]);
    zobs              = (float*)mxGetPr(prhs[5]);

    Nx_imgwidth     =   (int)mxGetScalar(prhs[6]);
    Ny_imgheight    =   (int)mxGetScalar(prhs[7]);
    delta_frequency = (float)mxGetScalar(prhs[8]);

    left   = (float)mxGetScalar(prhs[ 9]);
    right  = (float)mxGetScalar(prhs[10]);
    bottom = (float)mxGetScalar(prhs[11]);
    top    = (float)mxGetScalar(prhs[12]);

    /* Section 3.
     * Set up some intermediate values */

    /* Range profile dimensions */
    Npulses    = mxGetN(prhs[0]);
    Nrangebins = mxGetM(prhs[0]);
    
    if (nrhs <= 15) {
        min_eff_idx       = (float)mxGetScalar(prhs[13]);
        total_proj_length = (float)mxGetScalar(prhs[14]);
    }
    else {
        min_eff_idx = 0;
        total_proj_length = Nrangebins;
    }


    /* CUDA parameters
     * FIXME: these should only be preset if Matlab didn't specify them */
    device      = 0;
    blockwidth  = BLOCKWIDTH;
    blockheight = BLOCKHEIGHT;
    if (MEXDEBUG) {
        printf("WARNING: CUDA parameters not provided. Auto-selecting:\n"
                "device      %d\n"
                "blockwidth  %d\n"
                "blockheight %d\n", device, blockwidth, blockheight);
    }

    /* Various collection-specific constants */

    c__4_delta_freq = CLIGHT / (4.0f*delta_frequency);

    /* FIXME: this TOTALLY prevents variable start frequency!!!! */
    pi_4_f0__clight = PI*4.0f*extract_f0(start_frequencies, Npulses) / CLIGHT;
    convert_f0(start_frequencies, Npulses);
    


    /* Section 4.
     * Set up Matlab outputs */
    plhs[0] = mxCreateNumericMatrix(Ny_imgheight, Nx_imgwidth, 
            FLOAT_CLASS, mxCOMPLEX);
    host_output_image.real = (float*)mxGetPr(plhs[0]);
    host_output_image.imag = (float*)mxGetPi(plhs[0]);


    /* Section 5.
     * Call Hartley's GPU initialization & invokation code */
    run_bp(range_profiles, xobs, yobs, zobs, 
            aimpoint_ranges, 
            Npulses, Nrangebins, Nx_imgwidth, Ny_imgheight,
            blockwidth, blockheight,
            device,
            host_output_image,
            0, Ny_imgheight,
            c__4_delta_freq, pi_4_f0__clight,
            start_frequencies, left, right, bottom, top, min_eff_idx, total_proj_length);
            
    
    return;
}



void from_gpu_complex_to_bp_complex_split(float2 * data, bp_complex_split out, int size) {
	int i;
	for (i = 0; i < size; i++) {
		out.real[i] = data[i].x;
		out.imag[i] = data[i].y;
	}
}

float2 * format_complex_to_columns(bp_complex_split a, int width_orig, int height_orig) {
	float2 * out = (float2 *) malloc(width_orig * height_orig * sizeof(float2));
	int i, j;
	for (i = 0; i < height_orig; i++) {
		int origOffset = i * width_orig;
		for (j = 0; j < width_orig; j++) {
			int newOffset = j * height_orig;
			out[newOffset + i].x = a.real[origOffset + j];
			out[newOffset + i].y = a.imag[origOffset + j];
		}
	}
	return out;
}

float2 * format_complex(bp_complex_split a, int size) {
	float2 * out = (float2 *) malloc(size * sizeof(float2));
	int i;
	for (i = 0; i < size; i++) {
		out[i].x = a.real[i];
		out[i].y = a.imag[i];
	}
	return out;
}

float4 * format_x_y_z_r(float * x, float * y, float * z, float * r, int size) {
	float4 * out = (float4 *) malloc(size * sizeof(float4));
	int i;
	for (i = 0; i < size; i++) {
		out[i].x = x[i];
		out[i].y = y[i];
		out[i].z = z[i];
		out[i].w = r[i];
	}
	return out;
}



void run_bp(bp_complex_split phd, float * xObs, float * yObs, float * zObs, float * r,
	int my_num_phi, int my_proj_length, int nxout, int nyout, int image_chunk_width,
	int image_chunk_height, int device, bp_complex_split host_output_image,
	int start_output_index, int num_output_rows,
    float c__4_delta_freq, float pi_4_f0__clight, float * start_frequencies,
    float left, float right, float bottom, float top, 
    float min_eff_idx, float total_proj_length) {
	
	MY_CUDA_SAFE_CALL(cudaSetDevice(device));

#if ZEROCOPY
    MY_CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif

	int num_out_bytes = 2 * sizeof(float) * num_output_rows * nyout;
	float2 * out_image;


	// Set up platform data texture
	float4 * trans_tex_platform_info = format_x_y_z_r(xObs, yObs, zObs, r, my_num_phi);
	cudaChannelFormatDesc float4desc = cudaCreateChannelDesc<float4>();
	cudaArray* array_tex_platform_info;

	MY_CUDA_SAFE_CALL(cudaMallocArray( &array_tex_platform_info, &float4desc, 
        my_num_phi, 1));
	MY_CUDA_SAFE_CALL(cudaMemcpyToArray(array_tex_platform_info, 0, 0, 
        trans_tex_platform_info, my_num_phi * 4
		* sizeof(float), cudaMemcpyHostToDevice));

	tex_platform_info.addressMode[0] = cudaAddressModeClamp;
	tex_platform_info.addressMode[1] = cudaAddressModeClamp;
	tex_platform_info.filterMode = cudaFilterModePoint;
	tex_platform_info.normalized = false; // access with normalized texture coordinates

	MY_CUDA_SAFE_CALL(cudaBindTextureToArray(tex_platform_info, array_tex_platform_info, float4desc));

	// Set up input projections texture
	float2 * projections = format_complex_to_columns(phd, my_proj_length, my_num_phi);
	
	cudaChannelFormatDesc float2desc = cudaCreateChannelDesc<float2>();
	cudaArray* cu_proj;

	MY_CUDA_SAFE_CALL(cudaMallocArray( &cu_proj, &float2desc, my_num_phi, my_proj_length));
	MY_CUDA_SAFE_CALL(cudaMemcpyToArray(cu_proj, 0, 0, projections, my_num_phi
		* my_proj_length * 2 * sizeof(float), cudaMemcpyHostToDevice));

	tex_projections.addressMode[0] = cudaAddressModeClamp;
	tex_projections.addressMode[1] = cudaAddressModeClamp;
	tex_projections.filterMode = cudaFilterModeLinear;
	tex_projections.normalized = false; // access with normalized texture coordinates

	MY_CUDA_SAFE_CALL(cudaBindTextureToArray(tex_projections, cu_proj, float2desc));

	// Set up and run the kernel
	dim3 dimBlock(image_chunk_width, image_chunk_height, 1);
	dim3 dimGrid(nxout/image_chunk_width, num_output_rows/image_chunk_height);

	float delta_pixel_x = (right-left) / (nxout-1);
	float delta_pixel_y = (top-bottom) / (nyout-1);
	float r_start_pre = (c__4_delta_freq*(float)total_proj_length/((float)total_proj_length-1.0f));
    
    float * device_start_frequencies;
	MY_CUDA_SAFE_CALL(cudaMalloc((void**) &device_start_frequencies, sizeof(float)*my_num_phi));
    MY_CUDA_SAFE_CALL(cudaMemcpy(device_start_frequencies, start_frequencies, sizeof(float)*my_num_phi, cudaMemcpyHostToDevice));


    clock_t c0, c1; 
    c0 = clock();
        
    float * debug_1, * debug_2, *debug_3, *debug_4;

#if ZEROCOPY
	MY_CUDA_SAFE_CALL(cudaHostAlloc((void**) &out_image, num_out_bytes, 
        cudaHostAllocMapped));

    float2 * device_pointer;
    MY_CUDA_SAFE_CALL(cudaHostGetDevicePointer((void **)&device_pointer, 
        (void *)out_image, 0));
    
	backprojection_loop<<<dimGrid, dimBlock>>>(device_pointer, my_num_phi, nyout, delta_pixel_x, delta_pixel_y,
		r_start_pre, total_proj_length, 0, start_output_index,
        c__4_delta_freq, device_start_frequencies, left, bottom, min_eff_idx, trans_tex_platform_info, 
        debug_1,debug_2,debug_3,debug_4,
        0,0);
#else

	MY_CUDA_SAFE_CALL(cudaMalloc((void**) &out_image, num_out_bytes));
	backprojection_loop<<<dimGrid, dimBlock>>>(out_image, my_num_phi, nyout, 
        delta_pixel_x, delta_pixel_y,
		r_start_pre, total_proj_length, 0, start_output_index,
        c__4_delta_freq, device_start_frequencies, left, bottom, min_eff_idx, trans_tex_platform_info,
        debug_1, debug_2,debug_3,debug_4,
        0,0);
#endif



    cudaError_t this_error = cudaGetLastError();
    if ( this_error != cudaSuccess) {
        printf("\nERROR: cudaGetLastError did NOT return success! DO NOT TRUST RESULTS!\n");
        printf("         '%s'\n", cudaGetErrorString(this_error) );
    }
 
    if ( cudaThreadSynchronize() != cudaSuccess)
        printf("\nERROR: threads did NOT synchronize! DO NOT TRUST RESULTS!\n\n");
    c1 = clock();
    printf("INFO: CUDA-mex kernel took %f s\n", (float) (c1 - c0)/CLOCKS_PER_SEC);

#if ZEROCOPY
	from_gpu_complex_to_bp_complex_split(out_image, host_output_image, num_output_rows * nyout);
    MY_CUDA_SAFE_CALL(cudaFreeHost(out_image));
#else
	float2 * host_data = (float2 *) malloc(num_out_bytes);
	//double start_t = -ms_walltime();
	MY_CUDA_SAFE_CALL(cudaMemcpy(host_data, out_image, num_out_bytes, cudaMemcpyDeviceToHost));
	//printf("MEMCPY,%lf\n", (start_t + ms_walltime()));
	from_gpu_complex_to_bp_complex_split(host_data, host_output_image, num_output_rows
		* nyout);
	free(host_data);
	cudaFree(out_image);
#endif
    cudaFree(device_start_frequencies);
	free(trans_tex_platform_info);
	free(projections);

	cudaFreeArray(array_tex_platform_info);
	cudaFreeArray(cu_proj);

    MY_CUDA_SAFE_CALL(cudaThreadExit());

}

__global__ void testing_platform_tex(float * x, float * y, float * z, float * w, float num)
{
    float4 foo = tex1D(tex_platform_info, num);
    x[0] = foo.x;
    y[0] = foo.y;
    z[0] = foo.z;
    w[0] = foo.w;
}

__global__ void testing_platform(float4 * plat, float * xx, float * yy, float * zz, float * ww, int num)
{
    float4 foo = plat[num];
    xx[0] = foo.x;
    yy[0] = foo.y;
    zz[0] = foo.z;
    ww[0] = foo.w;
}

__global__ void testing_proj_tex(float * re, float * im, float xx, float yy)
{
    float2 foo = tex2D(tex_projections, xx, yy); // x: proj num, y: rbin
    re[0] = foo.x;
    im[0] = foo.y;
}

__global__ void testing_r(float xpixel, float ypixel, float xa, float ya, float za, float * R) 
{
    (*R) = ( CAREFUL_AMINUSB_SQ(xpixel, xa) + CAREFUL_AMINUSB_SQ(ypixel, ya) + 
            __fmul_rn(za, za));
}



/* Main kernel.
 *
 * Tuning options:
 * - is it worth #defining radar parameters like start_frequency?
 *      ............  or imaging parameters like xmin/ymax?
 * - Make sure (4 pi / c) is computed at compile time!
 * - Use 24-bit integer multiplications!
 *
 * */
__global__ void backprojection_loop(float2 * full_image,
        int Nphi, int nyout, float delta_pixel_x, float delta_pixel_y, 
        float R_START_PRE, int PROJ_LENGTH,
        int X_OFFSET, int Y_OFFSET,
        float C__4_DELTA_FREQ, float  * PI_4_F0__CLIGHT, 
        float left, float bottom, float min_eff_idx, float4 * platform_info,
        float * debug_effective_idx, float * debug_2, float * x_mat, float * y_mat, 
        float rmin, float rmax) {

    float2 subimage;
    subimage = make_float2(0.0f, 0.0f);
    float2 csum; // For compensated sum
    float y, t;
    csum = make_float2(0.0f, 0.0f);

    float xpixel = left   + (float)(blockIdx.x * BLOCKWIDTH  + threadIdx.x) * 
        delta_pixel_x;
    float ypixel = bottom + (float)(blockIdx.y * BLOCKHEIGHT + threadIdx.y) * 
        delta_pixel_y;
    
    float2 texel;

    __shared__ int proj_num;
    __shared__ float4 platform;
    __shared__ int copyblock;

    __shared__ float delta_r;
    delta_r = rmax - rmin;
    __shared__ float Nl1_dr;
    Nl1_dr = __fdiv_rn((float)PROJ_LENGTH - 1.0f, delta_r);

    copyblock = (blockIdx.y * BLOCKHEIGHT) * nyout + blockIdx.x * BLOCKWIDTH;

    /* Now, let's loop through these projections! 
     * */
#pragma unroll 3
    for (proj_num=0; proj_num < Nphi; ++proj_num) {

        //platform = tex1D(tex_platform_info, (float)proj_num + 0.5f);
        platform = platform_info[proj_num];

        /* R_reciprocal = 1/R = 1/sqrt(sum_{# in xyz} [#pixel - #platform]^2),
         * This is the distance between the platform and every pixel.
         */
         /*
        float R = sqrtf( 
                (xpixel - platform.x) * 
                (xpixel - platform.x) +
                (ypixel - platform.y) * 
                (ypixel - platform.y) +
                platform.z * platform.z);*/
        float R = MAKERADIUS(xpixel, ypixel, platform.x, platform.y, platform.z);

        /* Per-pixel-projection phasor = exp(1j 4 pi/c * f_min * R). */
        //float2 pixel_scale = expjf_div_2(PI_4_F0__CLIGHT[proj_num] * R * 0.5f);
        float2 pixel_scale = expjf(PI_4_F0__CLIGHT[proj_num] * R);
        
        /* The fractional range bin for this pixel, this projection */
        /*
        float effective_idx = ((float)PROJ_LENGTH-1.0f) *
            (R - ( platform.w - R_START_PRE )) / (2.0f*C__4_DELTA_FREQ) 
            - min_eff_idx;*/
        //float effective_idx = ((float)PROJ_LENGTH-1.0f) / (rmax - rmin) * (R - platform.w - rmin);
        float effective_idx = __fmul_rn(Nl1_dr , __fadd_rn(__fadd_rn(R, -1.0f*platform.w), -1.0f*rmin));

        /* This is the interpolated range profile element for this pulse */

        // Flipped textres
        /*texel = tex2D(tex_projections, 
                0.5f+effective_idx, 0.5f+(float)proj_num);*/
        // offset textures
        texel = tex2D(tex_projections, 0.5f+(float)proj_num, 0.5f+effective_idx);

        /* Scale "texel" by "pixel_scale".
           The RHS of these 2 lines just implement complex multiplication.
        */
        y = REAL(texel)*REAL(pixel_scale) - REAL(csum);
        t = subimage.x + y;
        csum.x = (t-subimage.x) - y;
        subimage.x = t;

        y = -1.0f*IMAG(texel)*IMAG(pixel_scale) - REAL(csum);
        t = subimage.x + y;
        csum.x = (t-subimage.x) - y;
        subimage.x = t;

        y = REAL(texel)*IMAG(pixel_scale) - IMAG(csum);
        t = subimage.y + y;
        csum.y = (t-subimage.y) - y;
        subimage.y = t;

        y = IMAG(texel)*REAL(pixel_scale) - IMAG(csum);
        t = subimage.y + y;
        csum.y = (t-subimage.y) - y;
        subimage.y = t;

        /*
        subimage.x += REAL(texel)*REAL(pixel_scale) - 
                IMAG(texel)*IMAG(pixel_scale);
        subimage.y += REAL(texel)*IMAG(pixel_scale) + 
                IMAG(texel)*REAL(pixel_scale);
        */

        if (proj_num==0) {
            debug_effective_idx[copyblock + (threadIdx.y) * nyout + threadIdx.x] = effective_idx;
            debug_2[copyblock + (threadIdx.y) * nyout + threadIdx.x] = R;
            x_mat[copyblock + (threadIdx.y) * nyout + threadIdx.x] = platform.x;
            y_mat[copyblock + (threadIdx.y) * nyout + threadIdx.x] = platform.y;
        }
    }
    /* Copy this thread's pixel back to global memory */
    //full_image[(blockIdx.y * BLOCKHEIGHT + threadIdx.y) * nyout + 
    //    blockIdx.x * BLOCKWIDTH + threadIdx.x] = subimage;
    full_image[copyblock + (threadIdx.y) * nyout + threadIdx.x] = subimage;
}


/* Credits: from BackProjectionKernal.c: "originally by reinke".
 * Given a float X, returns float2 Y = exp(j * X).
 *
 * __device__ code is always inlined. */
__device__ 
float2 expjf(float in) {
    float2 out;
    float t, tb;
#if USE_FAST_MATH
    t = __tanf(in / 2.0f);
#else
    t = tan(in / 2.0f);
#endif
    tb = t*t + 1.0f;
    out.x = (2.0f - tb) / tb; /* Real */
    out.y = (2.0f * t) / tb; /* Imag */
    return out;
}

__device__ 
float2 expjf_div_2(float in) {
    float2 out;
    float t, tb;
    //t = __tanf(in - (float)((int)(in/(PI2)))*PI2 );
    t = __tanf(in - PI * rintf(in/PI) );
    tb = t*t + 1.0f;
    out.x = (2.0f - tb) / tb; /* Real */
    out.y = (2.0f * t) / tb; /* Imag */
    return out;
}


