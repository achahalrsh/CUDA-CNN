#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>
#include <fstream>
#include <math.h>
#include <cstring>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer
{
public:
	int M, N, O;
	bool is_gpu;

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	int *maxIndices;

	Layer(int M, int N, int O, bool gpu_flag);
	Layer(int M, int N, int O, FILE *weights_file, bool gpu_flag);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();
	void save(std::ofstream &weights_file);

	void cpu_setOutput(float *data);
	void cpu_clear();
	void cpu_bp_clear();
	void cpu_save(std::ofstream &weights_file);
};

// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]);
__global__ void fp_bias_c1(float preact[6][24][24], float bias[6]);
// __global__ void fp_maxpool_s1(float input[6][24][24], float output[6][6][6], int* maxIndices, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int poolSize, int channels);
__global__ void fp_avgpool_s1(float input[6][24][24], float preact[6][6][6]);
// __global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4]);
// __global__ void fp_bias_s1(float preact[6][6][6], float bias[1]);
__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6]);
__global__ void fp_bias_f(float preact[10], float bias[10]);


// Back propagation kernels
__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]);
__global__ void bp_bias_f(float bias[10], float d_preact[10]);
__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
// __global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]);
// __global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]);
// __global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
// __global__ void bp_maxpool_s1(float d_input_grad[6][24][24], float d_output_grad[6][6][6], int* maxIndices, int inputHeight, int inputWidth, int outputHeight, int outputWidth, int poolSize, int channels);
__global__ void bp_avgpool_s1(float d_output[6][24][24], float nd_preact[6][6][6]);
__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]);



// Utility CPU functions
float cpu_step_function(float v);
void cpu_apply_step_function(float *input, float *output, int N);
void cpu_makeError(float *err, float *output, unsigned int Y, const int N);
void cpu_apply_grad(float *output, float *grad, const int N);

// CPU forward propagation
void cpu_fp_preact_c1(float input[28][28],float preact[6][24][24], float weight[6][5][5]);
void cpu_fp_bias_c1(float preact[6][24][24], const float bias[6]);
void cpu_fp_avgpool_s1(const float input[6][24][24], float preact[6][6][6]);
void cpu_fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6]);
void cpu_fp_bias_f(float preact[10], const float bias[10]);

// CPU backward propagation
void cpu_bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]);
void cpu_bp_bias_f(float bias[10], float d_preact[10]);
void cpu_bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
void cpu_bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
void cpu_bp_avgpool_s1(float d_output[6][24][24], float nd_preact[6][6][6]);
void cpu_bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
void cpu_bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
void cpu_bp_bias_c1(float bias[6], float d_preact[6][24][24]);