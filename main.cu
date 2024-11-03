#include <iostream>
#include <random>
#include <algorithm>
#include <filesystem>

#include "data.h"
#include "maxk_forward.h"

using namespace std;

string base_dir = "./graphs/";

#define DIM_MUL_N 1
#define DIM_MUL(x) ((x + DIM_MUL_N - 1) / DIM_MUL_N) * DIM_MUL_N

void test_graph(string graph) {

    int dim_origin = 256;
    int dim_k_list[] = {16, 32, 64, 96, 128, 192};
    int dim_k_limit = 64;

    int *cu_indptr, *cu_indices;
    int v_num = cuda_read_array(&cu_indptr, base_dir + graph + ".indptr") - 1;
    int e_num = cuda_read_array(&cu_indices, base_dir + graph + ".indices");

    cout << "Vnum =  " << v_num << endl;
    cout << "Enum =  " << e_num << endl;

    float *cu_val;
    cudaError_t status;

    status = cudaMallocManaged(&cu_val, e_num * sizeof(float));

    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed for cu_val: %s\n", cudaGetErrorString(status));
        return;
    }

    float *cu_vout_ref;
    float *cu_vin_sparse, *cu_vin_sparse_data, *cu_vout_maxk, *cu_vout_maxk_backward;
    u_int8_t *cu_vin_sparse_selector;

    cudaMallocManaged(&cu_vout_ref, v_num * dim_origin * sizeof(float));
    cudaMallocManaged(&cu_vin_sparse, v_num * dim_origin * sizeof(float));
    cudaMallocManaged(&cu_vin_sparse_data, v_num * DIM_MUL(dim_k_limit) * sizeof(float));
    cudaMallocManaged(&cu_vout_maxk_backward, v_num * DIM_MUL(dim_k_limit) * sizeof(float));
    cudaMallocManaged(&cu_vin_sparse_selector, v_num * DIM_MUL(dim_k_limit) * sizeof(u_int8_t));
    cudaMallocManaged(&cu_vout_maxk, v_num * dim_origin * sizeof(float));

    default_random_engine engine;
    engine.seed(123);
    uniform_real_distribution<float> rd(0, 1);

    generate(cu_val, cu_val + e_num, [&](){ return rd(engine); });
    generate(cu_vin_sparse, cu_vin_sparse + v_num * dim_origin, [&]() { return rd(engine); });
    generate(cu_vin_sparse_data, cu_vin_sparse_data + v_num * dim_k_limit, [&]() { return rd(engine); });

    vector<int> sequence(dim_origin);
    iota(sequence.begin(), sequence.end(), 0);

    MaxkForward forward(graph, cu_indptr, cu_indices, cu_val, cu_vin_sparse_data, cu_vout_maxk, v_num, e_num, dim_origin);
    double t_maxk = forward.do_test(true, dim_origin);

}

int main(int argc, char *argv[]) {

    string arg_graph(argv[1]);
    test_graph(arg_graph);

    return 0;
}
