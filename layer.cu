#include "layer.h"


/* 
0. CPU version of the code
1. Pooling layer
2. Timing for each section (load, forward, error, backward)
3. Inference timings
4. Simplify fp_preact function etc (indexing in 2D/3D etc).
5. Try different architectures.
*/

// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i)
	{
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < M; ++j)
		{
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMalloc(&maxIndices, sizeof(int) * O);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Constructor
Layer::Layer(int M, int N, int O, bool cpu)
{
	this->M = M;
	this->N = N;
	this->O = O;

	output = new float[O];
	preact = new float[O];
	bias = new float[N];
	weight = new float[M*N];
	d_output = new float[O];
	d_preact = new float[O];
	d_weight = new float[M*N];
	maxIndices = new int[O];

	// Initialize biases and weights
	for (int i = 0; i < N; ++i) {
		bias[i] = 0.5f - static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		
		for (int j = 0; j < M; ++j) {
			weight[i * M + j] = 0.5f - static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		}
	}
}

// Constructor
Layer::Layer(int M, int N, int O, FILE *weights_file, bool cpu)
{
	this->M = M;
	this->N = N;
	this->O = O;

	output = new float[O];
	preact = new float[O];
	bias = new float[N];
	weight = new float[M*N];
	d_output = new float[O];
	d_preact = new float[O];
	d_weight = new float[M*N];
	maxIndices = new int[O];

	// Initialize biases and weights
    for (int i = 0; i < N; ++i)
	{
		fscanf(weights_file, "%f", &bias[i]);

		for (int j = 0; j < M; ++j)
		{
			fscanf(weights_file, "%f", &weight[i * M + j]);
		}
	}
}

// Constructor (using weights from file)
Layer::Layer(int M, int N, int O, FILE *weights_file)
{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i)
	{
		fscanf(weights_file, "%f", &h_bias[i]);

		for (int j = 0; j < M; ++j)
		{
			fscanf(weights_file, "%f", &h_weight[i][j]);
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);
	cudaMalloc(&maxIndices, sizeof(int) * O);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
	cudaFree(maxIndices);

	delete[] output;
	delete[] preact;
	delete[] bias;
	delete[] weight;
	delete[] d_output;
	delete[] d_preact;
	delete[] d_weight;
	delete[] maxIndices;

}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

void Layer::cpu_setOutput(float *data) {
    for (int i = 0; i < O; ++i) {
        output[i] = data[i];
    }
}



// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::cpu_clear()
{
	memset(output, 0, sizeof(float) * O);
	memset(preact, 0, sizeof(float) * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}

void Layer::cpu_bp_clear()
{
    memset(d_output, 0, sizeof(float) * O);
    memset(d_preact, 0, sizeof(float) * O);
    memset(d_weight, 0, sizeof(float) * M * N);
}


void Layer::save(std::ofstream &weights_file)
{
	float h_bias[N];
	float h_weight[N][M];

	cudaMemcpy(h_bias, bias, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_weight, weight, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i)
	{
		weights_file << std::fixed << h_bias[i] << "\n";

		for (int j = 0; j < M; ++j)
		{
			weights_file << std::fixed << h_weight[i][j] << "\n";
		}
	}
}

__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

float cpu_step_function(float v)
{
	return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx)
	{
		output[idx] = step_function(input[idx]);
	}
}

void cpu_apply_step_function(float *input, float *output, int N) {
    for (int idx = 0; idx < N; ++idx) {
        output[idx] = cpu_step_function(input[idx]);
    }
}


__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx)
	{
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

void cpu_makeError(float *err, float *output, unsigned int Y, int N) {
    for (int idx = 0; idx < N; ++idx) {
        err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
    }
}


__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx)
	{
		output[idx] += dt * grad[idx];
	}
}

void cpu_apply_grad(float *output, float *grad, int N) {
    for (int idx = 0; idx < N; ++idx) {
        output[idx] += dt * grad[idx];
    }
}


__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 5 * 5 * 6 * 24 * 24;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 5);
		const int i2 = ((idx /= 5) % 5);
		const int i3 = ((idx /= 5) % 6);
		const int i4 = ((idx /= 6) % 24);
		const int i5 = ((idx /= 24) % 24);

		atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
	}
}

void cpu_fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]) {
    const int N = 5 * 5 * 6 * 24 * 24;

    // Initialize preact to zero
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
            for (int k = 0; k < 24; ++k) {
                preact[i][j][k] = 0;
            }
        }
    }

    for (int n = 0; n < N; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1) % 5);
        const int i2 = ((idx /= 5) % 5);
        const int i3 = ((idx /= 5) % 6);
        const int i4 = ((idx /= 6) % 24);
        const int i5 = ((idx /= 24) % 24);

        // Ensure indices are within bounds
        if ((i4 + i1 < 28) && (i5 + i2 < 28)) {
            preact[i3][i4][i5] += weight[i3][i1][i2] * input[i4 + i1][i5 + i2];
        }
    }
}



__global__ void fp_bias_c1(float preact[6][24][24], float bias[6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 24 * 24;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 24);
		const int i3 = ((idx /= 24) % 24);

		preact[i1][i2][i3] += bias[i1];
	}
}

void cpu_fp_bias_c1(float preact[6][24][24], const float bias[6]) {
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
            for (int k = 0; k < 24; ++k) {
                preact[i][j][k] += bias[i];
            }
        }
    }
}

// __global__ void fp_maxpool_s1(float input[6][24][24], float output[6][6][6], int maxIndices[6*6*6], int inputWidth, int inputHeight, int outputWidth, int outputHeight, int poolSize, int channels) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int z = blockIdx.z * blockDim.z + threadIdx.z;

//     if (x < outputWidth && y < outputHeight && z < channels) {
//         float maxVal = -INFINITY;
//         int maxIndex = -1;
//         for (int i = 0; i < poolSize; ++i) {
//             for (int j = 0; j < poolSize; ++j) {
//                 int inputX = x + i;
//                 int inputY = y + j;
//                 if (inputX < inputWidth && inputY < inputHeight) {
//                     float val = input[z][inputY][inputX];
//                     if (val > maxVal) {
//                         maxVal = val;
//                         maxIndex = inputY * inputWidth + inputX;
//                     }
//                 }
//             }
//         }
//         output[z][y][x] = maxVal;
//         maxIndices[z * outputWidth * outputHeight + y * outputWidth + x] = maxIndex;
//     }
// }

__global__ void fp_avgpool_s1(float input[6][24][24], float preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4 * 4 * 6 * 6 * 6;
	const float d = pow(4.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 4);
		const int i2 = ((idx /= 4) % 4);
		const int i3 = ((idx /= 4) % 6);
		const int i4 = ((idx /= 6) % 6);
		const int i5 = ((idx /= 6) % 6);

		atomicAdd(&preact[i3][i4][i5], input[i3][i4 * 4 + i1][i5 * 4 + i2]/d);
	}
}

void cpu_fp_avgpool_s1(const float input[6][24][24], float preact[6][6][6]) {
    const int N = 4 * 4 * 6 * 6 * 6;
    const float d = 4.0f * 4.0f;  // The area of the pooling window

    // Initialize preact to zero
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                preact[i][j][k] = 0;
            }
        }
    }

    for (int n = 0; n < N; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1) % 4);
        const int i2 = ((idx /= 4) % 4);
        const int i3 = ((idx /= 4) % 6);
        const int i4 = ((idx /= 6) % 6);
        const int i5 = ((idx /= 6) % 6);

        preact[i3][i4][i5] += input[i3][i4 * 4 + i1][i5 * 4 + i2] / d;
    }
}


// __global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
// {
// 	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
// 	const int size = blockDim.x * gridDim.x;

// 	const int N = 4 * 4 * 6 * 6 * 6;

// 	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
// 	{
// 		int idx = n;
// 		const int i1 = ((idx /= 1) % 4);
// 		const int i2 = ((idx /= 4) % 4);
// 		const int i3 = ((idx /= 4) % 6);
// 		const int i4 = ((idx /= 6) % 6);
// 		const int i5 = ((idx /= 6) % 6);

// 		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
// 	}
// }

// __global__ void fp_bias_s1(float preact[6][6][6], float bias[1])
// {
// 	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
// 	const int size = blockDim.x * gridDim.x;

// 	const int N = 6 * 6 * 6;

// 	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
// 	{
// 		int idx = n;
// 		const int i1 = ((idx /= 1) % 6);
// 		const int i2 = ((idx /= 6) % 6);
// 		const int i3 = ((idx /= 6) % 6);

// 		preact[i1][i2][i3] += bias[0];
// 	}
// }

__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 10);
		const int i2 = ((idx /= 10) % 6);
		const int i3 = ((idx /= 6) % 6);
		const int i4 = ((idx /= 6) % 6);

		atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
	}
}

void cpu_fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6]) {
    const int N = 10 * 6 * 6 * 6;

    // Initialize preact to zero
    for (int i = 0; i < 10; ++i) {
        preact[i] = 0;
    }

    for (int n = 0; n < N; ++n) {
        int idx = n;
        const int i1 = ((idx /= 1) % 10);
        const int i2 = ((idx /= 10) % 6);
        const int i3 = ((idx /= 6) % 6);
        const int i4 = ((idx /= 6) % 6);

        preact[i1] += weight[i1][i2][i3][i4] * input[i2][i3][i4];
    }
}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx)
	{
		preact[idx] += bias[idx];
	}
}

void cpu_fp_bias_f(float preact[10], const float bias[10]) {
    const int N = 10;

    for (int idx = 0; idx < N; ++idx) {
        preact[idx] += bias[idx];
    }
}


__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 10);
		const int i2 = ((idx /= 10) % 6);
		const int i3 = ((idx /= 6) % 6);
		const int i4 = ((idx /= 6) % 6);

		d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
	}
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos + 1) / size; ++idx)
	{
		bias[idx] += dt * d_preact[idx];
	}
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 10);
		const int i2 = ((idx /= 10) % 6);
		const int i3 = ((idx /= 6) % 6);
		const int i4 = ((idx /= 6) % 6);

		atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
	}
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 6);
		const int i3 = ((idx /= 6) % 6);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

// __global__ void bp_maxpool_s1(float d_input_grad[6][24][24], float d_output_grad[6][6][6], int maxIndices[6*6*6], int inputHeight, int inputWidth, int outputHeight, int outputWidth, int poolSize, int channels) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int z = blockIdx.z * blockDim.z + threadIdx.z;

//     if (x < outputWidth && y < outputHeight && z < channels) {
//         int index = z * outputWidth * outputHeight + y * outputWidth + x;
//         int maxIndex = maxIndices[index];

//         // Assuming the maxIndices are stored as linear indices, we convert them back to 2D indices for the input.
//         int maxIndexY = maxIndex / inputWidth;
//         int maxIndexX = maxIndex % inputWidth;

//         // Propagate the gradient only through the path of the maximum value.
//         d_input_grad[z][maxIndexY][maxIndexX] = d_output_grad[z][y][x];
//     }
// }

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1 * 4 * 4 * 6 * 6 * 6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 1);
		const int i2 = ((idx /= 1) % 4);
		const int i3 = ((idx /= 4) % 4);
		const int i4 = ((idx /= 4) % 6);
		const int i5 = ((idx /= 6) % 6);
		const int i6 = ((idx /= 6) % 6);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 6 * 6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 6);
		const int i3 = ((idx /= 6) % 6);

		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
	}
}

__global__ void bp_avgpool_s1(float d_output[6][24][24], float nd_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1 * 4 * 4 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 1);
		const int i2 = ((idx /= 1) % 4);
		const int i3 = ((idx /= 4) % 4);
		const int i4 = ((idx /= 4) % 6);
		const int i5 = ((idx /= 6) % 6);
		const int i6 = ((idx /= 6) % 6);

		atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], nd_preact[i4][i5][i6]);
	}
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1 * 4 * 4 * 6 * 6 * 6;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 1);
		const int i2 = ((idx /= 1) % 4);
		const int i3 = ((idx /= 4) % 4);
		const int i4 = ((idx /= 4) % 6);
		const int i5 = ((idx /= 6) % 6);
		const int i6 = ((idx /= 6) % 6);

		atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
	}
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 24 * 24;

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 24);
		const int i3 = ((idx /= 24) % 24);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 5 * 5 * 24 * 24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 5);
		const int i3 = ((idx /= 5) % 5);
		const int i4 = ((idx /= 5) % 24);
		const int i5 = ((idx /= 24) % 24);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
	}
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6 * 24 * 24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
	{
		int idx = n;
		const int i1 = ((idx /= 1) % 6);
		const int i2 = ((idx /= 6) % 24);
		const int i3 = ((idx /= 24) % 24);

		atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
	}
}