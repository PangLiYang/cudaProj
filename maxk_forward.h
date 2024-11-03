#pragma once
#include "base.h"

class MaxkForward : public Base {
public:
    using Base::Base;

    float *vout_ref;
    u_int8_t *vin_sparse_selector;
    int dim_sparse;

    double do_test(bool timing, int dim);

protected:
    int *_warp4;
    int num_warps;

    void run(int dim);
};