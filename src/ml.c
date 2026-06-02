#include "ml.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Tensor* tensor_create(int rows, int cols) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->rows = rows;
    t->cols = cols;
    t->data = (float*)calloc(rows * cols, sizeof(float));
    return t;
}

void tensor_free(Tensor* t) {
    if (t) {
        free(t->data);
        free(t);
    }
}

void tensor_fill(Tensor* t, float val) {
    for (int i = 0; i < t->rows * t->cols; i++) {
        t->data[i] = val;
    }
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (a->rows != b->rows || a->cols != b->cols) return NULL;
    Tensor* out = tensor_create(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (a->cols != b->rows) return NULL;
    Tensor* out = tensor_create(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            out->data[i * out->cols + j] = sum;
        }
    }
    return out;
}

void tensor_relu(Tensor* t) {
    for (int i = 0; i < t->rows * t->cols; i++) {
        if (t->data[i] < 0) t->data[i] = 0;
    }
}

void tensor_sigmoid(Tensor* t) {
    for (int i = 0; i < t->rows * t->cols; i++) {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
}

void tensor_print(Tensor* t) {
    printf("Tensor(%d, %d):\n", t->rows, t->cols);
    for (int i = 0; i < t->rows; i++) {
        for (int j = 0; j < t->cols; j++) {
            printf("%.4f ", t->data[i * t->cols + j]);
        }
        printf("\n");
    }
}
