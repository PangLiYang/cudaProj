#include <string>
#include <iostream>

#include "maxk_forward.h"
#include "data.h"

using namespace std;

extern string base_dir, graph;

const int WARPS_PER_BLOCK = 12;
const int EXT_WARP_DIM = 32;

#define DIM_MUL_N 1
#define DIM_MUL(x) ((x + DIM_MUL_N - 1) / DIM_MUL_N) * DIM_MUL_N

__global__ void forwardKernel (
        const int *,
        const int *,
        const float *,
        const float *,
        const u_int8_t *,
        float *,
        const int,
        const int,
        const int,
        const int,
        const int
);

void MaxkForward::run(int dim) {

    int sharedMem_size = WARPS_PER_BLOCK * dim * sizeof(float);
    forwardKernel <<< grid, block, sharedMem_size >>> (_warp4, idx, val, vin, vin_sparse_selector, vout, num_v, num_e, dim, dim_sparse, num_warps);
}

double MaxkForward::do_test(bool timing, int dim) {

    this->num_warps = cuda_read_array(&this->_warp4, "../w" + to_string(WARPS_PER_BLOCK) + "_nz" + "64_warp_4/" + this->_graph + ".warp4") / 4;
    int block_num = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (!timing) {
        cout << "block num = " << block_num << endl;
    }

    grid.x = block_num;
    block.x = WARPS_PER_BLOCK * EXT_WARP_DIM;

    double ret = timing_body(timing, dim);

    cudaFree(this->_warp4);
    return ret;
}

__global__ void forwardKernel(const int *_warp4, const int *idx, const float *val, const float *vin_data, const u_int8_t *vin_selector, float *vout, const int num_v, const int num_e, const int feat_in, const int dim_sparse, const int num_warps) {

    const int4 *warp4 = reinterpret_cast<const int4 *>(_warp4);
    extern __shared__ float out_cache[];

    const int tId = blockIdx.x * blockDim.x + threadIdx.x;
    const int warpId = tId / EXT_WARP_DIM;
    const int laneId = threadIdx.x / EXT_WARP_DIM;
    const int wId = threadIdx.x % EXT_WARP_DIM;

    const int sparse_laneId = laneId % dim_sparse;
    const int sparse_wid = wId * (EXT_WARP_DIM / dim_sparse) + laneId / dim_sparse;

    int4 sparse_wInfo, wInfo;
    int warpRow, warpLoc, warpLen;
    int sparse_warpRow, sparse_warpLoc, sparse_warpLen;

    if (warpId < num_warps) {
        wInfo = warp4[warpId];
        warpRow = wInfo.x;
        warpLoc = wInfo.y;
        warpLen = wInfo.z;

        if (dim_sparse < 32 && blockIdx.x * blockDim.x / EXT_WARP_DIM + sparse_wid < num_warps) {
            sparse_wInfo = warp4[blockIdx.x * blockDim.x / EXT_WARP_DIM + sparse_wid];
            sparse_warpRow = wInfo.x;
            sparse_warpLoc = wInfo.y;
            sparse_warpLen = wInfo.z;
        }
    }

    #pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext += 1) {
        out_cache[threadIdx.x + ext * blockDim.x] = 0;
    }

    if (warpId >= num_warps || blockIdx.x * blockDim.x / EXT_WARP_DIM + sparse_wid >= num_warps) {
        return;
    }

    __syncthreads();

    if (dim_sparse < 32) {

        if (sparse_wid < blockDim.x / EXT_WARP_DIM && laneId / dim_sparse < EXT_WARP_DIM / dim_sparse) {

            for (int i = 0; i < sparse_warpLen; i += 1) {
                int nzLoc = sparse_warpLoc + i;

                float leftV = __ldg(val + nzLoc);
                int rightLoc = __ldg(val + nzLoc); * DIM_MUL(dim_sparse) + sparse_laneId;
                float rightV = vin_data[rightLoc];

                out_cache[sparse_wid * feat_in + vin_selector[rightLoc]] += leftV * rightV;
            }

        }

        __syncthreads();

    } else {

        for (int i = 0; i < warpLen; i += 1) {

            for (int j = laneId; j < dim_sparse; j += 1) {

                int nzLoc = warpLoc + i;

                float leftV = __ldg(val + nzLoc);
                int rightLoc = __ldg(val + nzLoc); * DIM_MUL(dim_sparse) + sparse_laneId;
                float rightV = vin_data[rightLoc];

                out_cache[wId * feat_in + vin_selector[rightLoc]] += leftV * rightV;
            }

        }

        __syncthreads();
    }

    #pragma unroll
    for (int ext = 0; ext < (feat_in + EXT_WARP_DIM - 1) / EXT_WARP_DIM; ext += 1) {

        atomicAdd(&vout[warpRow * feat_in + laneId + ext * EXT_WARP_DIM], out_cache[wId * feat_in + laneId + ext * EXT_WARP_DIM]);
    }

}