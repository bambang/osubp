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
        float min_eff_idx);

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

