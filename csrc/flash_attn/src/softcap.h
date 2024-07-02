/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/fast_math.h>

#include "utils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
    for (int i=0; i < size<0>(tensor); ++i){  // MMA
        for (int mi; mi < size<1>(tensor); ++mi){
            for (int nj; nj < size<2>(tensor); ++nj){
                float tmp_val = fast_tanh(tensor(i, mi, nj) / softcap) * softcap
                tensor(i, mi, nj) = tmp_val
            }
        }
    }
}

}