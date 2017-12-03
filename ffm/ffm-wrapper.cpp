/*
The following table is the meaning of some variables in this code:

W: The pointer to the beginning of the model
w: Dynamic pointer to access values in the model
m: Number of fields
k: Number of latent factors
n: Number of features
l: Number of data points
f: Field index (0 to m-1)
d: Latent factor index (0 to k-1)
j: Feature index (0 to n-1)
i: Data point index (0 to l-1)
nnz: Number of non-zero elements
X, P: Used to store the problem in a compressed sparse row (CSR) format. len(X) = nnz, len(P) = l + 1
Y: The label. len(Y) = l
R: Precomputed scaling factor to make the 2-norm of each instance to be 1. len(R) = l
v: Value of each element in the problem
*/

#pragma GCC diagnostic ignored "-Wunused-result" 
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <new>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <cassert>
#include <numeric>

#define DEBUG 0

#if defined USESSE
#include <pmmintrin.h>
#endif

#if defined USEOMP
#include <omp.h>
#endif


#include "ffm-wrapper.h"
#include "ffm.cpp"

namespace ffm {

ffm_problem* ffm_convert_data(ffm_line* data, ffm_int num_lines) {
    ffm_float* Y = new ffm_float[num_lines];
    ffm_float* R = new ffm_float[num_lines];
    ffm_long* P = new ffm_long[num_lines + 1];
    P[0] = 0;

    ffm_long num_nodes = 0;

    ffm_line *data_begin = data;
    ffm_line *data_end = data + num_lines;

    for (ffm_line *line = data_begin; line != data_end; line++) {
        num_nodes = num_nodes + line->size;
    }

    ffm_node* X = new ffm_node[num_nodes];
    int m = 0;
    int n = 0;

    ffm_long p = 0;
    ffm_int i = 0;
    for (ffm_line *line = data_begin; line != data_end; line++) {
        ffm_float y = line->label > 0 ? 1.0f : -1.0f;
        ffm_float scale = 0;

        ffm_node* node_beg = line->data;
        ffm_node* node_end = node_beg + line->size;

        for (ffm_node* N = node_beg; N != node_end; N++) {
            X[p] = *N;

            m = max(m, N->f + 1);
            n = max(n, N->j + 1);

            scale += N->v * N->v;
            p++;
        }

        Y[i] = y;
        R[i] = 1.0 / scale;
        P[i + 1] = p;
        i++;
    }

    ffm_problem* result = new ffm_problem;
    #if defined DEBUG
    //printf("allocated address in ffm_convert_data: %p\n", result);
    # endif

    result->size = num_lines;

    result->data = X;
    result->num_nodes = num_nodes;
    result->pos = P;

    result->labels = Y;
    result->scales = R;
    result->n = n;
    result->m = m;

    return result;
}

void free_ffm_data(ffm_problem *data) {
    delete data->data;
    delete data->labels;
    delete data->pos;
    delete data->scales;
    delete data;
}

ffm_model ffm_init_model(ffm_problem& problem, ffm_parameter params) {
    int n = problem.n;
    int m = problem.m;
    return init_model(n, m, params);
}

ffm_float ffm_train_iteration(ffm_problem& prob, ffm_model& model, ffm_parameter params) {
    ffm_double loss = 0;

    ffm_int len = prob.size;
    ffm_node* X = prob.data;
    ffm_float* Y = prob.labels;
    ffm_float* R = prob.scales;

    ffm_long* P = prob.pos;

    ffm_int* idx = new ffm_int[len];
    for (int i = 0; i < len; i++) {
        idx[i] = i;
    }

    random_shuffle(&idx[0], &idx[len]);

    #if defined USEOMP
    #pragma omp parallel for schedule(static) reduction(+: loss)
    #endif

    for (ffm_int id = 0; id < len; id++) {
        ffm_int i = idx[id];
        ffm_float y = Y[i];

        ffm_node *begin = &X[P[i]];
        ffm_node *end = &X[P[i + 1]];

        ffm_float r = params.normalization ? R[i] : 1;
        ffm_float t = wTx(begin, end, r, model);

        ffm_float expnyt = exp(-y * t);
        loss = loss + log(1 + expnyt);

        ffm_float kappa = -y * expnyt / (1 + expnyt);
        wTx(begin, end, r, model, kappa, params.eta, params.lambda, true);
    }

    delete[] idx;

    return loss / len;
}

ffm_float* ffm_predict_batch(ffm_problem &prob, ffm_model &model) {
    ffm_node* X = prob.data;
    ffm_float* R = prob.scales;
    ffm_long* P = prob.pos;
    ffm_int len = prob.size;

    ffm_float* result = new float[len];

    for (ffm_int i = 0; i < len; i++) {
        ffm_node *begin = &X[P[i]];
        ffm_node *end = &X[P[i + 1]];

        ffm_float r = model.normalization ? R[i] : 1.0;
        ffm_float t = wTx(begin, end, r, model);

        result[i] = 1 / (1 + exp(-t));
    }

    #if defined DEBUG
    //printf("allocated address in ffm_predict_batch: %p\n", result);
    #endif
    
    return result;
}

void ffm_save_model_c_string(ffm_model& model, char* path) {
    string str_path(path);
    ffm_save_model(model, str_path);
}

ffm_model ffm_load_model_c_string(char* path) {
    string str_path(path);
    return ffm_load_model(str_path);
}

ffm_float ffm_predict_array(ffm_node* nodes, int len, ffm_model &model) {
    ffm_node* begin = nodes;
    ffm_node* end = begin + len;

    return ffm_predict(begin, end, model);
}

void free_ffm_float(ffm_float *data) {
    #if defined DEBUG
    //printf("freeing ffm_float address: %p\n", data);
    # endif
    delete data;
}

void free_ffm_problem(ffm_problem *data) {
    #if defined DEBUG
    //printf("freeing ffm_problem address: %p\n", data);
    # endif
    delete data->data;
    delete data->labels;
    delete data->pos;
    delete data->scales;
    delete data;
}


} // namespace ffm
