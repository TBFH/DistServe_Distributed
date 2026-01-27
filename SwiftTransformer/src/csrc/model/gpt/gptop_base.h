#pragma once

#include <string>

#include <torch/script.h>

#include "model/gpt/gpt_base.h"
#include "model/gpt/gpt.h"

namespace st::model {

// Please refer to gpt_base.h for the design of GPTBase, Gpt, and GptOpBase.
class GptOpBase : public torch::CustomClassHolder {
public:
	GptBase* gpt;	// A pointer to GptBase, which can be Gpt<T> for any T.
	bool weight_loaded;

	GptOpBase(
		std::string inference_dtype,
		GptHyperParam hyper_param,
		GptPagedAttnParam pagedattn_param,
		GptParallelismParam parallelism_param
	);

	~GptOpBase();

	void loadWeight(const std::string& weight_path);
	void initDummyWeight();

    std::vector<int64_t> forward(
        const std::vector<std::vector<int64_t>> &input_tokens_batched,
	    const std::vector<int64_t> &first_token_indexes,
        torch::Tensor &k_cache,
        torch::Tensor &v_cache,
        const std::vector<std::vector<int64_t>> &block_table,
		torch::Tensor &intermed_input,
        torch::Tensor &intermed_output,

		torch::Tensor &current_layer_input,
		const int64_t layer_id,
		const int64_t operation
    );

	void init_communicator(const std::vector<int64_t> tp_id, const std::vector<int64_t> pp_id);
};

}