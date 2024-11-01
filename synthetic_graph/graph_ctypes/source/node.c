#include <stdlib.h>
#include "node.h"

double* run_node(Node *node) {
    double **input_results = (double **)malloc(node->input_count * sizeof(double *));

    for (int i = 0; i < node->input_count; i++) {
        input_results[i] = run_node(node->inputs[i]);
    }
    double *result = node->fcn(input_results, node->input_sizes, node->input_count, node->data);

    for (int i = 0; i < node->input_count; i++) {
        if (!node->inputs[i]->data) {
            free(input_results[i]);
        }
    }
    free(input_results);

    return result;
}