"""
Adapted from https://github.com/vllm/worker/worker.py
"""
import copy
import time
from typing import List, Tuple, Optional
import socket

import ray
import torch
import torch.distributed

from distserve.config import ModelConfig, CacheConfig, ParallelConfig
from distserve.request import Request, BatchedRequests
from distserve.utils import set_random_seed, cudaMemoryIpcHandle, Stage
from distserve.models import get_model_op
from distserve.utils import get_gpu_memory, set_random_seed, GB, MB
from distserve.logger import init_logger
from distserve.downloader import download_and_convert_weights

logger = init_logger(__name__)


# If we call `torch.ops.swapping_ops.swap` in `ParaWorker.swap_blocks()` directly,
# it will result in a "cannot pickle" error. Don't know why
def call_swapping_op(
    source_block_ids: List[int],
    target_block_ids: List[int],
    is_swap_in: bool,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_swap: torch.Tensor,
    v_swap: torch.Tensor,
):
    """Call the swapping operation."""
    # The swapping operation is a custom C++ operator that swaps the blocks
    # between the CPU and GPU. The operator is defined in
    # FastServe/fastserve/swapping_ops.cpp.
    torch.ops.swapping_ops.swap(
        source_block_ids,
        target_block_ids,
        is_swap_in,
        k_cache,
        v_cache,
        k_swap,
        v_swap,
    )


@ray.remote(num_cpus=0, num_gpus=1)
class ParaWorker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache, the KV swap and executing the model on the GPU.
    In case of distributed inference, each worker is assigned a partition of
    the model.

    """

    def __init__(
        self,
        worker_id: int,
        stage: Stage,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig = ParallelConfig(),
        tensor_parallel_id: List[int] = None,   # Although the type is list[int], it is actually a NCCL unique ID
        pipeline_parallel_id: List[int] = None, # Same as above
    ) -> None:
        self.worker_id = worker_id
        self.stage = stage
        self.model = None
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.tensor_parallel_id = tensor_parallel_id
        self.pipeline_parallel_id = pipeline_parallel_id
        self.gpu_id = ray.get_gpu_ids()[0]
        
        self.device = torch.device(f"cuda:0")
        torch.cuda.set_device(self.device)
        
        # K/V cache on GPU
        self.k_cache = None
        self.v_cache = None
        # K/V swap on CPU
        self.k_swap = None
        self.v_swap = None
        # CUDA streams for swapping in and out
        self.swap_in_stream = torch.cuda.Stream()
        self.swap_out_stream = torch.cuda.Stream()
        # The swap_event_table, refer to block_manager.py for more details
        self.swap_event_table = {}
        # The latest swap event in each stream
        # Used when we need to wait for all swap events to finish
        self.latest_swap_in_event = None
        self.latest_swap_out_event = None
        # Statistics
        self.execution_time = 0.0
        self.blocked_swapping_time = 0.0
        # Intermediate results buffer for pipeline_parallel
        self.intermed_input = None
        self.intermed_output = None

    def ready(self):
        """
        Ray functions queue inside one single actor to be executed in order.
        If ready is called, the actor is ready.
        """
        logger.info(f"Worker {self.stage}.#{self.worker_id} created on host {socket.gethostname()} and gpu #{self.gpu_id}")
        pass

    def init_model(self):
        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model_op(
            self.model_config, self.parallel_config, self.cache_config
        )
        self.model.init_communicator(self.tensor_parallel_id, self.pipeline_parallel_id)
        torch.cuda.synchronize()
        if self.model_config.use_dummy_weights:
            self.model.init_dummy_weights()
        else:
            path = download_and_convert_weights(self.model_config)
            self.model.load_weight(path)
        torch.cuda.synchronize()
        logger.info(f"(worker {self.stage}.#{self.worker_id}) model {self.model_config.model} loaded")

    def init_kvcache_and_swap(self, num_gpu_blocks, num_cpu_blocks):
        """
        Allocate the K/V cache and swap.
        
        Return K/V cache's memory handle
        """
        # kv shape is [num_gpu_blocks, num_layers, num_local_heads, block_size, head_dim]
        # profile the GPU to get num_gpu_blocks
        kv_cache_shape = (
            num_gpu_blocks,
            self.model_config.get_num_layers(self.parallel_config),
            self.model_config.get_num_heads(self.parallel_config),
            self.cache_config.block_size,
            self.model_config.get_head_size(),
        )
        self.k_cache = torch.empty(
            kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        self.v_cache = torch.empty(
            kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        # kv swap is [num_cpu_blocks, num_layers, num_local_heads, block_size, head_dim]
        # We pin memory here in order to leverage cudaMemcpyAsync when swapping
        kv_swap_shape = (num_cpu_blocks,) + kv_cache_shape[1:]
        self.k_swap = torch.empty(
            kv_swap_shape, dtype=self.model_config.get_torch_dtype(), device="cpu", pin_memory=True
        )
        self.v_swap = torch.empty(
            kv_swap_shape, dtype=self.model_config.get_torch_dtype(), device="cpu", pin_memory=True
        )
        torch.cuda.synchronize()

    def _get_block_size_in_bytes(
        self,
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        # the shape of one slot in k/v cache is [num_layers, num_local_heads, block_size, head_dim]
        num_layers = model_config.get_num_layers(parallel_config)
        num_heads = model_config.get_num_heads(parallel_config)
        head_dim = model_config.get_head_size()

        key_cache_size = num_layers * num_heads * block_size * head_dim
        total = key_cache_size * 2
        dtype_size = model_config.get_dtype_size()
        return total * dtype_size

    @torch.inference_mode()
    def _profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # GPU and CPU blocks that can be allocated with the remaining free memory.

        # Profile memory usage with max_batch_size requests and the total
        # number of tokens equal to max_tokens_per_batch.
        total_gpu_memory = get_gpu_memory()
        peak_runtime_memory = (
            total_gpu_memory * 0.01
            + self.model_config.get_model_size_in_bytes(
                parallel_config=self.parallel_config
            )
        )
        logger.info(f"runtime peak memory: {peak_runtime_memory / GB:.3f} GB")
        logger.info(f"total GPU memory: {total_gpu_memory / GB:.3f} GB")
        block_size_in_bytes = self._get_block_size_in_bytes(
            block_size, self.model_config, self.parallel_config
        )
        logger.info(
            f"kv cache size for one token: {block_size_in_bytes / block_size / MB:.5f} MB"
        )
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_runtime_memory)
            // block_size_in_bytes
        )
        num_cpu_blocks = int(cpu_swap_space // block_size_in_bytes)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        logger.info(f"num_gpu_blocks: {num_gpu_blocks}")
        num_cpu_blocks = max(num_cpu_blocks, 0)
        logger.info(f"num_cpu_blocks: {num_cpu_blocks}")

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        # return num_gpu_blocks, num_cpu_blocks
        return 150, 1

    def step(
        self,
        request_ids: List[int],
        input_tokens_batched,
        first_token_indexes,
        block_table,
        
        current_layer_input,
        layer_id,
        operation,
        intermed = None,
    ) -> List[int]:
        """Run one step of inference on the batch of requests."""

        start = time.time()
        # Check whether synchronization is necessary
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # We let the current stream wait for the swap event
                # This is non-blocking (It just stop the current stream instead
                # of chocking the CPU)
                self.swap_event_table[request_id].wait(torch.cuda.current_stream())
                self.swap_event_table.pop(request_id, None)
        self.blocked_swapping_time += time.time() - start

        intermed_shape = (
            sum([len(req) for req in input_tokens_batched]),
            self.model_config.get_hidden_size()
        )
        self.intermed_input = torch.empty(
            0, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        self.intermed_output = torch.empty(
            intermed_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        # if not self.parallel_config.is_first_stage() and len(input_tokens_batched) > 0:
        #     _, inter_in = intermed
        #     self.intermed_input = inter_in
        _, inter_in = intermed
        if inter_in != None:
            self.intermed_input = inter_in

        start = time.time()
        # print(f"Worker {self.stage}.#{self.worker_id} Step begin")
        # run forward
        generated_tokens_ids = self.model.forward(
            input_tokens_batched,
            first_token_indexes,
            self.k_cache,
            self.v_cache,
            block_table,
            self.intermed_input,
            self.intermed_output,

            current_layer_input,
            layer_id,
            operation
        )
        self.execution_time += time.time() - start
        # print(f"Worker {self.stage}.#{self.worker_id} Step end")

        return generated_tokens_ids, copy.deepcopy(self.intermed_output)
    
    def send_kvcache(
        self,
        source_block_indexes: List[int],
        layer_bound: Tuple[int, int],
        head_bound: Tuple[int, int]
    ):
        kcache_to_migrate = []
        vcache_to_migrate = []
        for idx in source_block_indexes:
            kcache_to_migrate.append(self.k_cache[idx][layer_bound[0]: layer_bound[1]][head_bound[0]: head_bound[1]])
            vcache_to_migrate.append(self.v_cache[idx][layer_bound[0]: layer_bound[1]][head_bound[0]: head_bound[1]])
        # return copy.deepcopy(kcache_to_migrate), copy.deepcopy(vcache_to_migrate)
        return kcache_to_migrate, vcache_to_migrate

    def receive_kvcache(
        self,
        target_block_indexes: List[int],
        layer_bound: Tuple[int, int],
        head_bound: Tuple[int, int],
        remote_context_kvcache
    ):
        k_cache, v_cache = remote_context_kvcache
        # print(f"\033[1;35m remote_context_kvcache got: {k_cache[0].shape} \n decode k_cache: {self.k_cache.shape} \033[0m")
        for i in range(len(k_cache)):
            self.k_cache[target_block_indexes[i]][layer_bound[0]: layer_bound[1]][head_bound[0]: head_bound[1]].copy_(k_cache[i])
            self.v_cache[target_block_indexes[i]][layer_bound[0]: layer_bound[1]][head_bound[0]: head_bound[1]].copy_(v_cache[i])
        return True
        
    def swap_blocks(
        self,
        request_ids: List[int],
        source_block_ids: List[int],
        target_block_ids: List[int],
        is_swap_in: bool,
    ):
        """Swap some blocks between CPU and GPU
        If is_swap_in, then move blocks from CPU to GPU, i.e. CPU block
        #source_block_ids[0] will be copied to GPU block #target_block_ids[0]
        and so on. Similar for is_swap_in = False
        """

        # print(f"Swap {source_block_ids} ({'CPU' if is_swap_in else 'GPU'}) to {target_block_ids} ({'GPU' if is_swap_in else 'CPU'})")
        stream = self.swap_in_stream if is_swap_in else self.swap_out_stream

        # Record event
        event = torch.cuda.Event()
        event.record(stream)

        # Save that event
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # If we've issued another swapping operation before, we shall wait it
                # Pay attention to the difference between wait() and synchronize()
                self.swap_event_table[request_id].wait(stream)
            self.swap_event_table[request_id] = event
        if is_swap_in:
            self.latest_swap_in_event = event
        else:
            self.latest_swap_out_event = event

        # Swap
        with torch.cuda.stream(stream):
            # torch.ops.swapping_ops.swap(
            #     source_block_ids,
            #     target_block_ids,
            #     is_swap_in,
            #     self.k_cache,
            #     self.v_cache,
            #     self.k_swap,
            #     self.v_swap,
            # )
            call_swapping_op(
                source_block_ids,
                target_block_ids,
                is_swap_in,
                self.k_cache,
                self.v_cache,
                self.k_swap,
                self.v_swap,
            )

    def clear_request_resource(self, request_id: int):
        """Clear the resources associated with the request."""
        """This is called by LLMEngine when a request is finished or aborted"""
        # Clear the swap event table
        self.swap_event_table.pop(request_id, None)

    def clear_request_resource_batched(self, requests: List[Request]):
        """Clear the resources associated with the requests."""
        for request in requests:
            self.clear_request_resource(request.request_id)

    def wait_for_all_swap_in(self):
        """Wait for all swap in to finish"""
        if self.latest_swap_in_event is not None:
            self.latest_swap_in_event.synchronize()
            self.latest_swap_in_event = None

    def wait_for_all_swap_out(self):
        """Wait for all swap out to finish"""
        if self.latest_swap_out_event is not None:
            self.latest_swap_out_event.synchronize()
            self.latest_swap_out_event = None

    def get_parallel_config(self):
        return self.parallel_config
