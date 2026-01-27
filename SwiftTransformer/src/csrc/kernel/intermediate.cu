#include "intermediate.h"

#include <cassert>
#include <cstdio>

#include "util/cuda_utils.h"

namespace st::kernel {

#define WARP_SIZE 32


// Tuneable parameters
constexpr int64_t DEFAULT_THREAD_BLOCK_SIZE = 256;

template<typename T>
__global__ void setIntermedKernel(
    T* __restrict__ target_data,
    const T* __restrict__ source_data,
    const int64_t numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        target_data[idx] = source_data[idx];
    }
    __syncthreads();
}

template<typename T>
void setIntermediateResult(
	const T* __restrict__ from,
	T* __restrict__ to,
	const int64_t num_tokens,
	const int64_t hidden_size
) {
    int64_t total_elements = num_tokens * hidden_size;
    int64_t n_blocks = (total_elements + DEFAULT_THREAD_BLOCK_SIZE - 1) / DEFAULT_THREAD_BLOCK_SIZE;
    setIntermedKernel<T><<<n_blocks, DEFAULT_THREAD_BLOCK_SIZE>>>(to, from, total_elements);
}

#define INSTANTIATE_SET_INTERMEDIATE_RESULT(T) \
    template void setIntermediateResult( \
        const T* __restrict__, \
        T* __restrict__, \
        const int64_t, \
        const int64_t \
    );

INSTANTIATE_SET_INTERMEDIATE_RESULT(float)
INSTANTIATE_SET_INTERMEDIATE_RESULT(half)

} // namespace st::kernel