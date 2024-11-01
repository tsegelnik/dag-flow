#ifndef NODE
#define NODE

typedef struct Node {
    struct Node** inputs;                           // Массив указателей на инпуты
    int *input_sizes;                     // Размеры входных массивов (превычисления)
    int input_count;                        // Количество входов (превычисления)
    double *(*fcn)(double **, int *, int, double *);  // Указатель на функцию
    double *data;                           // Данные для инпута
} Node;

double* run_node(Node *node);

#endif