#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <iostream>
#include <cuda.h>
#include <cstdio>
#include <time.h>
#include <assert.h>
#include <limits>
#include <cmath> 

using namespace std;

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer *l_input;
static Layer *l_c1;
static Layer *l_s1;
static Layer *l_f;

static double learn(int start_epoch, int end_epoch);
static double cpu_learn(int start_epoch, int end_epoch);
static unsigned int classify(double data[28][28]);
static unsigned int cpu_classify(double data[28][28]);
static double test();
static double cpu_test();
static double forward_pass(double data[28][28]);
static double cpu_forward_pass(double data[28][28]);
static double back_pass();
static double cpu_back_pass();

static double time_to_load_data = 0.0;
static double time_to_init_layers = 0.0;
static double time_to_destroy_layers = 0.0;
static double time_fwd_pass = 0.0;
static double time_conv = 0.0;
static double time_pooling = 0.0;
static double time_feedfwd = 0.0;
static double time_bwd_pass = 0.0;
static double time_conv_bwd = 0.0;
static double time_pooling_bwd = 0.0;
static double time_feedbwd = 0.0;
static double time_to_save_weights = 0.0;
static double time_train = 0.0;
static double time_test = 0.0;

static void printTimings() {
	cout << "Total train time: " << time_train << endl;
	cout << "Total test time: " << time_test << endl;
	cout << "Total dataload time: " << time_to_load_data << endl;
	cout << "Total layer init time: " << time_to_init_layers << endl;
	//Forward layer
	cout << "Total forward pass time: " << time_fwd_pass << endl;
	cout << "Fwd convolution time: " << time_conv << endl;
	cout << "Fwd poolin time: " << time_pooling << endl;
	cout << "Final feed fwd layer time: " << time_feedfwd << endl;
	//Backward layer
	cout << "Total backward pass time: " << time_bwd_pass << endl;
	cout << "Bwd convolution time: " << time_conv_bwd << endl;
	cout << "Bwd poolin time: " << time_pooling_bwd << endl;
	cout << "Final feed bwd layer time: " << time_feedbwd << endl;
	// Save Weights
	cout << "Save weights time: " << time_to_save_weights << endl;
}

static double loaddata()
{
	clock_t start, end;
	start = clock();
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
						 &train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
						 &test_set, &test_cnt);

	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static double init_layers()
{
	clock_t start, end;
	start = clock();
	l_input = new Layer(0, 0, 28 * 28, true);
	l_c1 = new Layer(5 * 5, 6, 24 * 24 * 6, true);
	l_s1 = new Layer(4 * 4, 1, 6 * 6 * 6, true);
	l_f = new Layer(6 * 6 * 6, 10, 10, true);
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static double cpu_init_layers()
{
	clock_t start, end;
	start = clock();
	l_input = new Layer(0, 0, 28 * 28, false);
	l_c1 = new Layer(5 * 5, 6, 24 * 24 * 6, false);
	l_s1 = new Layer(4 * 4, 1, 6 * 6 * 6, false);
	l_f = new Layer(6 * 6 * 6, 10, 10, false);
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static double cpu_init_layers(const char *weights_file)
{
	clock_t start, end;
	start = clock();
	FILE *file = fopen(weights_file, "r");
	l_input = new Layer(0, 0, 28 * 28, file, false);
	l_c1 = new Layer(5 * 5, 6, 24 * 24 * 6, file, false);
	l_s1 = new Layer(4 * 4, 1, 6 * 6 * 6, file, false);
	l_f = new Layer(6 * 6 * 6, 10, 10, file, false);
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static double init_layers(const char *weights_file)
{
	clock_t start, end;
	start = clock();
	FILE *file = fopen(weights_file, "r");
	l_input = new Layer(0, 0, 28 * 28, file, true);
	l_c1 = new Layer(5 * 5, 6, 24 * 24 * 6, file, true);
	l_s1 = new Layer(4 * 4, 1, 6 * 6 * 6, file, true);
	l_f = new Layer(6 * 6 * 6, 10, 10, file, true);
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static double save_weights(const char *weights_file)
{
	clock_t start, end;
	start = clock();
	std::ofstream file(weights_file);
	l_input->save(file);
	l_c1->save(file);
	l_s1->save(file);
	l_f->save(file);
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static double destroy_layers()
{
	clock_t start, end;
	start = clock();
	delete l_input;
	delete l_c1;
	delete l_s1;
	delete l_f;
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static float computeL2Norm(float* vec, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}



int main(int argc, const char **argv)
{
	assert(argc > 1 && "Run with mode -both, -train, -train-increment or -test");
	srand(time(NULL));

	if (strcmp(argv[1], "-train") == 0)
	{
		assert(argc == 4 && "Please provide the number of epochs and weights file");
		time_to_load_data = loaddata();
		time_to_init_layers = init_layers();
		time_train = learn(1, atoi(argv[2]));
		time_to_save_weights = save_weights(argv[3]);
		time_to_destroy_layers = destroy_layers();
		printTimings();
	}
	else if (strcmp(argv[1], "-test") == 0)
	{
		assert(argc == 3 && "Please provide the weights file");
		time_to_load_data = loaddata();
		time_to_init_layers = init_layers(argv[2]);
		time_test = test();
		time_to_destroy_layers = destroy_layers();
		printTimings();
	}
	else if (strcmp(argv[1], "-both") == 0)
	{
		time_to_load_data = loaddata();
		time_to_init_layers = init_layers();
		time_train = learn(1, 5);
		time_test = test();
		time_to_destroy_layers = destroy_layers();
		printTimings();
	}
	else if (strcmp(argv[1], "-cputrain") == 0) {
		time_to_load_data = loaddata();
		time_to_init_layers = cpu_init_layers();
		time_train = cpu_learn(1, atoi(argv[2]));
		time_to_save_weights = save_weights(argv[3]);
		time_to_destroy_layers = destroy_layers();
		printTimings();
	}
	else if (strcmp(argv[1], "-cputest") == 0) {
		time_to_load_data = loaddata();
		time_to_init_layers = cpu_init_layers(argv[2]);
		time_test = cpu_test();
		time_to_destroy_layers = destroy_layers();
		printTimings();
	}
	else
	{
		assert(0 && "Run with mode -both, -train or -test");
	}

	return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			input[i][j] = data[i][j];
		}
	}

	l_input->clear();
	l_c1->clear();
	l_s1->clear();
	l_f->clear();

	clock_t start, start_pool, start_feed_fwd;
	start = clock();

	l_input->setOutput((float *)input);


	fp_preact_c1<<<64, 64>>>((float(*)[28])l_input->output, (float(*)[24][24])l_c1->preact, (float(*)[5][5])l_c1->weight);
	fp_bias_c1<<<64, 64>>>((float(*)[24][24])l_c1->preact, l_c1->bias);
	apply_step_function<<<64, 64>>>(l_c1->preact, l_c1->output, l_c1->O);

	time_conv += ((double)(clock() - start)) / CLOCKS_PER_SEC;

	// dim3 blockSize(8, 8, 8); // Block size of 8x8x8
	// // For fp_maxpool_s1 (output size of 6x6x6)
	// dim3 gridSizeFp((6 + blockSize.x - 1) / blockSize.x, (6 + blockSize.y - 1) / blockSize.y, (6 + blockSize.z - 1) / blockSize.z);
	// fp_maxpool_s1<<<gridSizeFp, blockSize>>>((float(*)[24][24])l_c1->output, (float(*)[6][6])l_s1->preact, (int*) l_s1->maxIndices, 24, 24, 6, 6, 4, 6);
	// apply_step_function<<<64, 64>>>(l_s1->preact, l_s1->output, l_s1->O);

	// Avergae Pool
	start_pool = clock();
	fp_avgpool_s1<<<64, 64>>>((float(*)[24][24])l_c1->output, (float(*)[6][6])l_s1->preact);
	apply_step_function<<<64, 64>>>(l_s1->preact, l_s1->output, l_s1->O);
	time_pooling += ((double)(clock() - start_pool)) / CLOCKS_PER_SEC;

	// Original layer
	// fp_preact_s1<<<64, 64>>>((float(*)[24][24])l_c1->output, (float(*)[6][6])l_s1->preact, (float(*)[4][4])l_s1->weight);
	// fp_bias_s1<<<64, 64>>>((float(*)[6][6])l_s1->preact, l_s1->bias);
	// apply_step_function<<<64, 64>>>(l_s1->preact, l_s1->output, l_s1->O);

	start_feed_fwd = clock();
	fp_preact_f<<<64, 64>>>((float(*)[6][6])l_s1->output, l_f->preact, (float(*)[6][6][6])l_f->weight);
	fp_bias_f<<<64, 64>>>(l_f->preact, l_f->bias);
	apply_step_function<<<64, 64>>>(l_f->preact, l_f->output, l_f->O);
	time_feedfwd += ((double)(clock() - start_feed_fwd)) / CLOCKS_PER_SEC;

	return ((double)(clock() - start)) / CLOCKS_PER_SEC;
}

// Forward propagation of a single row in dataset
static double cpu_forward_pass(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			input[i][j] = data[i][j];
		}
	}

	l_input->cpu_clear();
	l_c1->cpu_clear();
	l_s1->cpu_clear();
	l_f->cpu_clear();

	clock_t start, start_pool, start_feed_fwd;
	start = clock();

	l_input->cpu_setOutput((float *)input);
	cpu_fp_preact_c1((float(*)[28])l_input->output, (float(*)[24][24])l_c1->preact, (float(*)[5][5])l_c1->weight);
	cpu_fp_bias_c1((float(*)[24][24])l_c1->preact, l_c1->bias);
	cpu_apply_step_function(l_c1->preact, l_c1->output, l_c1->O);
	time_conv += ((double)(clock() - start)) / CLOCKS_PER_SEC;

	// Avergae Pool
	start_pool = clock();
	cpu_fp_avgpool_s1((float(*)[24][24])l_c1->output, (float(*)[6][6])l_s1->preact);
	cpu_apply_step_function(l_s1->preact, l_s1->output, l_s1->O);

	time_pooling += ((double)(clock() - start_pool)) / CLOCKS_PER_SEC;

	// Original layer
	// fp_preact_s1<<<64, 64>>>((float(*)[24][24])l_c1->output, (float(*)[6][6])l_s1->preact, (float(*)[4][4])l_s1->weight);
	// fp_bias_s1<<<64, 64>>>((float(*)[6][6])l_s1->preact, l_s1->bias);
	// apply_step_function<<<64, 64>>>(l_s1->preact, l_s1->output, l_s1->O);
	start_feed_fwd = clock();
	cpu_fp_preact_f((float(*)[6][6])l_s1->output, l_f->preact, (float(*)[6][6][6])l_f->weight);
	cpu_fp_bias_f(l_f->preact, l_f->bias);
	cpu_apply_step_function(l_f->preact, l_f->output, l_f->O);
	time_feedfwd += ((double)(clock() - start_feed_fwd)) / CLOCKS_PER_SEC;

	return ((double)(clock() - start)) / CLOCKS_PER_SEC;
}


// Back propagation to update weights
static double back_pass()
{
	clock_t start, start_pool, start_conv;

	start = clock();

	bp_weight_f<<<64, 64>>>((float(*)[6][6][6])l_f->d_weight, l_f->d_preact, (float(*)[6][6])l_s1->output);
	bp_bias_f<<<64, 64>>>(l_f->bias, l_f->d_preact);

	bp_output_s1<<<64, 64>>>((float(*)[6][6])l_s1->d_output, (float(*)[6][6][6])l_f->weight, l_f->d_preact);
	bp_preact_s1<<<64, 64>>>((float(*)[6][6])l_s1->d_preact, (float(*)[6][6])l_s1->d_output, (float(*)[6][6])l_s1->preact);
	time_feedbwd += ((double)(clock() - start)) / CLOCKS_PER_SEC;
	// Original s1 layer
	// bp_weight_s1<<<64, 64>>>((float(*)[4][4])l_s1->d_weight, (float(*)[6][6])l_s1->d_preact, (float(*)[24][24])l_c1->output);
	// bp_bias_s1<<<64, 64>>>(l_s1->bias, (float(*)[6][6])l_s1->d_preact);
	// bp_output_c1<<<64, 64>>>((float(*)[24][24])l_c1->d_output, (float(*)[4][4])l_s1->weight, (float(*)[6][6])l_s1->d_preact);

	// Average Pooling
	start_pool = clock();
	bp_avgpool_s1<<<64, 64>>>((float(*)[24][24])l_c1->d_output, (float(*)[6][6])l_s1->d_preact);
	time_pooling_bwd += ((double)(clock() - start_pool)) / CLOCKS_PER_SEC;
	// dim3 blockSize(8, 8, 8); // Block size of 8x8x8
	// // For bp_maxpool_s1 (input size of 6x24x24)
	// dim3 gridSizeBp((24 + blockSize.x - 1) / blockSize.x, (24 + blockSize.y - 1) / blockSize.y, (6 + blockSize.z - 1) / blockSize.z);

	// bp_maxpool_s1<<<gridSizeBp,blockSize>>>((float(*)[24][24])l_c1->d_output, (float(*)[6][6])l_s1->d_preact, (int*) l_s1->maxIndices, 24, 24, 6, 6, 4, 6);

	start_conv = clock();
	bp_preact_c1<<<64, 64>>>((float(*)[24][24])l_c1->d_preact, (float(*)[24][24])l_c1->d_output, (float(*)[24][24])l_c1->preact);
	bp_weight_c1<<<64, 64>>>((float(*)[5][5])l_c1->d_weight, (float(*)[24][24])l_c1->d_preact, (float(*)[28])l_input->output);
	bp_bias_c1<<<64, 64>>>(l_c1->bias, (float(*)[24][24])l_c1->d_preact);
	time_conv_bwd += ((double)(clock() - start_conv)) / CLOCKS_PER_SEC;

	apply_grad<<<64, 64>>>(l_f->weight, l_f->d_weight, l_f->M * l_f->N);
	apply_grad<<<64, 64>>>(l_s1->weight, l_s1->d_weight, l_s1->M * l_s1->N);
	apply_grad<<<64, 64>>>(l_c1->weight, l_c1->d_weight, l_c1->M * l_c1->N);

	return ((double)(clock() - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double cpu_back_pass()
{
	clock_t start, start_pool, start_conv;

	start = clock();

	cpu_bp_weight_f((float(*)[6][6][6])l_f->d_weight, l_f->d_preact, (float(*)[6][6])l_s1->output);
	cpu_bp_bias_f(l_f->bias, l_f->d_preact);

	cpu_bp_output_s1((float(*)[6][6])l_s1->d_output, (float(*)[6][6][6])l_f->weight, l_f->d_preact);
	cpu_bp_preact_s1((float(*)[6][6])l_s1->d_preact, (float(*)[6][6])l_s1->d_output, (float(*)[6][6])l_s1->preact);
	time_feedbwd += ((double)(clock() - start)) / CLOCKS_PER_SEC;
	// Original s1 layer
	// bp_weight_s1<<<64, 64>>>((float(*)[4][4])l_s1->d_weight, (float(*)[6][6])l_s1->d_preact, (float(*)[24][24])l_c1->output);
	// bp_bias_s1<<<64, 64>>>(l_s1->bias, (float(*)[6][6])l_s1->d_preact);
	// bp_output_c1<<<64, 64>>>((float(*)[24][24])l_c1->d_output, (float(*)[4][4])l_s1->weight, (float(*)[6][6])l_s1->d_preact);

	// Average Pooling
	start_pool = clock();
	cpu_bp_avgpool_s1((float(*)[24][24])l_c1->d_output, (float(*)[6][6])l_s1->d_preact);
	time_pooling_bwd += ((double)(clock() - start_pool)) / CLOCKS_PER_SEC;
	// dim3 blockSize(8, 8, 8); // Block size of 8x8x8
	// // For bp_maxpool_s1 (input size of 6x24x24)
	// dim3 gridSizeBp((24 + blockSize.x - 1) / blockSize.x, (24 + blockSize.y - 1) / blockSize.y, (6 + blockSize.z - 1) / blockSize.z);

	// bp_maxpool_s1<<<gridSizeBp,blockSize>>>((float(*)[24][24])l_c1->d_output, (float(*)[6][6])l_s1->d_preact, (int*) l_s1->maxIndices, 24, 24, 6, 6, 4, 6);
	start_conv = clock();
	cpu_bp_preact_c1((float(*)[24][24])l_c1->d_preact, (float(*)[24][24])l_c1->d_output, (float(*)[24][24])l_c1->preact);
	cpu_bp_weight_c1((float(*)[5][5])l_c1->d_weight, (float(*)[24][24])l_c1->d_preact, (float(*)[28])l_input->output);
	cpu_bp_bias_c1(l_c1->bias, (float(*)[24][24])l_c1->d_preact);
	time_conv_bwd += ((double)(clock() - start_conv)) / CLOCKS_PER_SEC;

	cpu_apply_grad(l_f->weight, l_f->d_weight, l_f->M * l_f->N);
	cpu_apply_grad(l_s1->weight, l_s1->d_weight, l_s1->M * l_s1->N);
	cpu_apply_grad(l_c1->weight, l_c1->d_weight, l_c1->M * l_c1->N);

	return ((double)(clock() - start)) / CLOCKS_PER_SEC;
}

static double learn(int start_epoch, int end_epoch)
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	clock_t start, end;

	start = clock();

	float err;

	double time_taken_fwd = 0.0;
	double time_taken_bwd = 0.0;

	fprintf(stdout, "Learning\n");

	for (int iter = start_epoch; iter <= end_epoch; iter++)
	{
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i)
		{
			float tmp_err;

			time_taken_fwd += forward_pass(train_set[i].data);

			l_f->bp_clear();
			l_s1->bp_clear();
			l_c1->bp_clear();

			// Euclid distance of train_set[i]
			makeError<<<10, 1>>>(l_f->d_preact, l_f->output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_f->d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken_bwd += back_pass();
		}

		err /= train_cnt;
		fprintf(stdout, "epoch: %d, error: %e, time_fwd_on_gpu: %lf\n, time_bwd_on_gpu: %lf\n", iter, err, time_taken_fwd, time_taken_bwd);

		if (err < threshold)
		{
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}
	}
	time_fwd_pass = time_taken_fwd;
	time_bwd_pass = time_taken_bwd;
	// fprintf(stdout, "\n Time - %lf\n", time_taken_fwd);
	// fprintf(stdout, "\n Time - %lf\n", time_taken_bwd);
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static double cpu_learn(int start_epoch, int end_epoch)
{
	float err;
	clock_t start, end;

	start = clock();
	double time_taken_fwd = 0.0;
	double time_taken_bwd = 0.0;

	fprintf(stdout, "Learning\n");

	for (int iter = start_epoch; iter <= end_epoch; iter++)
	{
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i)
		{
			float tmp_err;

			time_taken_fwd += cpu_forward_pass(train_set[i].data);

			l_f->cpu_bp_clear();
			l_s1->cpu_bp_clear();
			l_c1->cpu_bp_clear();

			// Euclid distance of train_set[i]
			cpu_makeError(l_f->d_preact, l_f->output, train_set[i].label, 10);
			// cout << l_f->d_preact;
			tmp_err = computeL2Norm(l_f->d_preact, 10);
			// cublasSnrm2(blas, 10, l_f->d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken_bwd += cpu_back_pass();
		}

		err /= train_cnt;
		fprintf(stdout, "epoch: %d, error: %e, time_fwd_on_cpu: %lf\n, time_bwd_on_cpu: %lf\n", iter, err, time_taken_fwd, time_taken_bwd);

		if (err < threshold)
		{
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}
	}
	time_fwd_pass = time_taken_fwd;
	time_bwd_pass = time_taken_bwd;
	// fprintf(stdout, "\n Time - %lf\n", time_taken_fwd);
	// fprintf(stdout, "\n Time - %lf\n", time_taken_bwd);
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	time_fwd_pass += forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f->output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i)
	{
		if (res[max] < res[i])
		{
			max = i;
		}
	}

	return max;
}
// Returns label of given data (0-9)
static unsigned int cpu_classify(double data[28][28])
{
	float res[10];

	time_fwd_pass += cpu_forward_pass(data);

	unsigned int max = 0;

	memcpy(res, l_f->output, sizeof(float) * 10);

	for (int i = 1; i < 10; ++i)
	{
		if (res[max] < res[i])
		{
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static double test()
{
	clock_t start, end;

	start = clock();
	int error = 0;

	for (int i = 0; i < test_cnt; ++i)
	{
		if (classify(test_set[i].data) != test_set[i].label)
		{
			++error;
		}
	}

	fprintf(stdout, "Test Accuracy: %.2lf%%\n",
					double(test_cnt - error) / double(test_cnt) * 100.0);
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// Perform forward propagation of test data
static double cpu_test()
{
	clock_t start, end;

	start = clock();
	int error = 0;

	for (int i = 0; i < test_cnt; ++i)
	{
		if (cpu_classify(test_set[i].data) != test_set[i].label)
		{
			++error;
		}
	}

	fprintf(stdout, "Test Accuracy: %.2lf%%\n",
					double(test_cnt - error) / double(test_cnt) * 100.0);
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}