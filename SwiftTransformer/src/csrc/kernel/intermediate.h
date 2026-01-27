#pragma once

namespace st::kernel {

template<typename T>
void setIntermediateResult(
	const T* __restrict__ from,
	T* __restrict__ to,
	const int64_t num_tokens,
	const int64_t hidden_size
);

}	// namespace st::kernel