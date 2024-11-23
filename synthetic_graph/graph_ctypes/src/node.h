#ifndef NODE
#define NODE

typedef struct Node {
    struct Node** inputs;
    int *input_sizes;
    int input_count;
    double *(*fcn)(double **, int *, int, double *);
    double *data;
} Node;

double* run_node(Node *node);

#endif