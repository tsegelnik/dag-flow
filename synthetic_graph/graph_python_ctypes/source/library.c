#include <stdlib.h>
#include <math.h>
#include "library.h"


double* input_fcn(double **lists, int *sizes, int num_lists, double *data) {
    return data;
}

double* sum_fcn(double **lists, int *sizes, int num_lists, double *data) {
    int size = sizes[0];

    for (int i = 0; i < num_lists; i++) {
        for (int j = 0; j < size; j++) {
            data[j] += lists[i][j];
        }
    }

    return data;
}

double* product_fcn(double **lists, int *sizes, int num_lists, double *data) {
    int size = sizes[0];

    for (int i = 0; i < num_lists; i++) {
        for (int j = 0; j < size; j++) {
            data[j] *= lists[i][j];
        }
    }

    return data;
}

double* integration_fcn(double **lists, int *sizes, int num_lists, double *data) {
    double *values = lists[0];
    double *bins = lists[1];
    for (int i = 0; i < sizes[0] - 1; i++) {
        data[0] += 0.5 * (values[i] + values[i + 1]) * (bins[i + 1] - bins[i]);
    }
    return data;
}

double* sin_fcn(double **lists, int *sizes, int num_lists, double *data) {
    double* input = lists[0];
    for (int i = 0; i < sizes[0]; i++) {
        data[i] = sin(input[i]);
    }
    return data;
}


double* cosh_fcn(double **lists, int *sizes, int num_lists, double *data) {
    double* input = lists[0];
    for (int i = 0; i < sizes[0]; i++) {
        data[i] = cosh(input[i]);
    }
    return data;
}
