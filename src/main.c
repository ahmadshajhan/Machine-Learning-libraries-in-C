#include "ml.h"
#include <stdio.h>

int main() {
    printf("c-ml-library: Initializing Neural Network Layers...\n\n");
    
    // Create weights and inputs
    Tensor* weights = tensor_create(2, 3);
    tensor_fill(weights, 0.5f);
    
    Tensor* inputs = tensor_create(3, 1);
    tensor_fill(inputs, 2.0f);
    
    // Forward pass
    Tensor* output = tensor_matmul(weights, inputs);
    
    // Apply activation
    tensor_relu(output);
    
    printf("Result of Forward Pass (MatMul + ReLU):\n");
    tensor_print(output);
    
    // Cleanup
    tensor_free(weights);
    tensor_free(inputs);
    tensor_free(output);
    
    return 0;
}
