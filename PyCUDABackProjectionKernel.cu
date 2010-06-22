/* vim: set tabstop=4, shiftwidth=4, expandtab */
#include "cuda.h"   /* CUDA */

/***
 * Compiler logics
 * **/
#define CLIGHT 299792458.0f        /* c: speed of light, m/s */
#define PI 3.141592653589793116f   /* pi, accurate to 128-bits */
#define PI2 6.283185307179586232f  /* 2*pi */
#define PI_4__CLIGHT (4.0f * PI / CLIGHT)

#define REAL(vec) (vec.x)
#define IMAG(vec) (vec.y)


/* The following macros are defined for common arithmetic operations. We use
 * the __f*_rn() CUDA intrinsics in order to prevent these operations from
 * being performed on FMADs, whose implementation forces the truncation of the
 * result between the multiple and add, giving us less accurate results.
*/
#define CAREFUL_AMINUSB_SQ(x,y) __fmul_rn(__fadd_rn((x), -1.0f*(y)), __fadd_rn((x), -1.0f*(y)))
#define MAKERADIUS(xpixel,ypixel, xa,ya,za) sqrtf(CAREFUL_AMINUSB_SQ(xpixel, xa) + CAREFUL_AMINUSB_SQ(ypixel, ya) + __fmul_rn(za, za))

#define BLOCKWIDTH    16
#define BLOCKHEIGHT   16


/***
 * Type defs
 * ***/
typedef float FloatType; /* FIXME: this should be used everywhere */


/***
 * Prototypes
 * ***/

__device__ float2 expjf(float in);
__device__ float2 expjf_div_2(float in);


/* Complex textures containing range profiles */
texture<float2, 2, cudaReadModeElementType> tex_projections;   


/* Main kernel.
 * */
__global__ void backprojection_loop(float2 * full_image,
        int Nphi, int IMG_HEIGHT, float delta_pixel_x, float delta_pixel_y, 
        int PROJ_LENGTH,
        float  * PI_4_F0__CLIGHT, 
        float LEFT, float BOTTOM, 
        float4 * PLATFORM_INFO,
        float rmin, float rmax) {

    float2 subimage;
    subimage = make_float2(0.0f, 0.0f);
    float2 csum; // For compensated sum
    float y, t;
    csum = make_float2(0.0f, 0.0f);

    float xpixel = LEFT + (float)(blockIdx.x * BLOCKWIDTH  + threadIdx.x) * 
        delta_pixel_x;
    float ypixel = BOTTOM + (float)(blockIdx.y * BLOCKHEIGHT + threadIdx.y) * 
        delta_pixel_y;
    
    float2 texel;

    __shared__ int proj_num;
    __shared__ float4 platform;
    __shared__ int copyblock;

    __shared__ float delta_r;
    delta_r = rmax - rmin;
    __shared__ float Nl1_dr;
    Nl1_dr = __fdiv_rn((float)PROJ_LENGTH - 1.0f, delta_r);

    copyblock = (blockIdx.y * BLOCKHEIGHT) * IMG_HEIGHT + blockIdx.x * BLOCKWIDTH;

    /* Now, let's loop through these projections! 
     * */
#pragma unroll 3
    for (proj_num=0; proj_num < Nphi; ++proj_num) {
        platform = PLATFORM_INFO[proj_num];

        /* R_reciprocal = 1/R = 1/sqrt(sum_{# in xyz} [#pixel - #platform]^2),
         * This is the distance between the platform and every pixel.
         */
         /* // This is what's *really* happening.
        float R = sqrtf( 
                (xpixel - platform.x) * 
                (xpixel - platform.x) +
                (ypixel - platform.y) * 
                (ypixel - platform.y) +
                platform.z * platform.z);*/
        float R = MAKERADIUS(xpixel, ypixel, platform.x, platform.y, platform.z);

        /* Per-pixel-projection phasor = exp(1j 4 pi/c * f_min * R). */
        // Faster, less accurate:
        // float2 pixel_scale = expjf_div_2(PI_4_F0__CLIGHT[proj_num] * R * 0.5f);
        // Slower, more accurate: 
        float2 pixel_scale = expjf(PI_4_F0__CLIGHT[proj_num] * R);
        
        /* The fractional range bin for this pixel, this projection. Implements
         * effective_idx = Nl1_dr*( R-platform.w-rmin ) 
         */
        float effective_idx = __fmul_rn(Nl1_dr , __fadd_rn(__fadd_rn(R, -1.0f*platform.w), -1.0f*rmin));

        /* This is the interpolated range profile element for this pulse */
        texel = tex2D(tex_projections, 0.5f+(float)proj_num, 0.5f+effective_idx);

        /* Scale "texel" by "pixel_scale".
           The RHS of these 2 lines just implement complex multiplication.
        */
        /*
        subimage.x += REAL(texel)*REAL(pixel_scale) - 
                IMAG(texel)*IMAG(pixel_scale);
        subimage.y += REAL(texel)*IMAG(pixel_scale) + 
                IMAG(texel)*REAL(pixel_scale);
        */

        // I'm experimenting with cumulative summing to try and reduce errors.
        // See http://en.wikipedia.org/wiki/Kahan_summation_algorithm
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

    }
    /* Copy this thread's pixel back to global memory */
    //full_image[(blockIdx.y * BLOCKHEIGHT + threadIdx.y) * IMG_HEIGHT + 
    //    blockIdx.x * BLOCKWIDTH + threadIdx.x] = subimage;
    full_image[copyblock + (threadIdx.y) * IMG_HEIGHT + threadIdx.x] = subimage;
}


/* Credits: from BackProjectionKernal.c: "originally by reinke".
 * Given a float X, returns float2 Y = exp(j * X).
 *
 * __device__ code is always inlined. */
__device__ 
float2 expjf(float in) {
    float2 out;
    float t, tb;
    t = tan(in / 2.0f);
    tb = t*t + 1.0f;
    out.x = (2.0f - tb) / tb; /* Real */
    out.y = (2.0f * t) / tb; /* Imag */
    return out;
}


/* This is derived from the above function, but is way faster because it uses
 * hardware __tanf(). This is allowable only if the argument to it is
 * "small"---within +/- 2\pi. 
 *
 * Given float X, returns exp(j * 2*X).
*/
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


