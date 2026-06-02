#ifndef C_ML_H
#define C_ML_H

#include <stddef.h>

// A basic Tensor structure for our ML library
typedef struct {
    int rows;
    int cols;
    float *data;
} Tensor;

// Core functions
Tensor* tensor_create(int rows, int cols);
void tensor_free(Tensor* t);
void tensor_fill(Tensor* t, float val);
void tensor_print(Tensor* t);

// Math operations
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_matmul(Tensor* a, Tensor* b);

// Activations
void tensor_relu(Tensor* t);
void tensor_sigmoid(Tensor* t);

#endif // C_ML_H
