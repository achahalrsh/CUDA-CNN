# CNN on CUDA
Implementation of Convolutional Neural Network using CUDA. On testing with MNIST dataset for 5 epochs, accuracy of 95% was obtained with a GPU training time of about 23 seconds.


### Compiling and Execution
To compile just navigate to root and nvcc *.cu -lcubas -o CNN
Executable can be run using `./CNN`
