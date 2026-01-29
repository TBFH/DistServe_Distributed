import time
import copy
from typing import List, Optional, Tuple, Dict, AsyncGenerator
import asyncio
import math
import argparse

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from distserve.config import (
    ModelConfig, 
    DisaggParallelConfig, 
    ParallelConfig, 
    CacheConfig, 
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.logger import init_logger
from distserve.request import (
    SamplingParams,
    Request,
    create_request,
)
from distserve.tokenizer import get_tokenizer
from distserve.utils import Counter
from distserve.single_stage_engine import (
    StepOutput,
    ContextStageLLMEngine,
    DecodingStageLLMEngine
)
from distserve.lifetime import LifetimeEvent, LifetimeEventType

logger = init_logger(__name__)

# 配置相关环境变量，防止通信问题导致的程序卡死
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] = 'eno4'


@ray.remote(num_gpus=1)
def resource_inspect():
    # GPU Overall Inspect
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    device_name = device.name()
    context = device.make_context()
    total_memory = device.total_memory() / (1024 ** 2)
    free_memory = cuda.mem_get_info()[0] / (1024 ** 2)
    used_memory = total_memory - free_memory
    context.pop()
    return {
        "GPU_Name": device_name,
        "Total_VRAM": total_memory,
        "Used_VRAM": used_memory,
        "Free_VRAM": free_memory,
    }


class LLMEngine:
    """
    LLMEngine: An LLMEngine launches the model executor workers and maintains runtime information.

    ## Overview

    This class, LLMEngine, receives requests from upper wrapper class and provides
    interface LLMEngine.generate() that yields the generated tokens for each request.

    It supports the feature of "disaggregate", which basically means to run 
    the context stage and the decoding stage on different GPUs to avoid interference.

    ## Implementation

    First let's inspect the automaton of one request:

            After
            context
            stage        |-------------|
    Waiting --------> Decoding <-------| After one decoding stage
                         |
                         |
                         V
                      Finished

    This class is implemented based on queues and event loops. There are three
    queues, two for scheduling and one for communication between event loops:
      - The waiting queue, maintained inside the ContextStageScheduler, which
        contains all the requests that are waiting for processing.
      - The decoding queue, maintained inside the DecodingStageScheduler, which
        contains all the requests that need further decoding.
      - The "bridge" queue, which contains all the requests that have just finished
        the context stage but have not been accepted by the decoding stage.
        (Producer: context stage event loop, Consumer: decoding stage event loop)
      
    Two event loops are executed concurrently and endlessly:
      - Context stage event loop. This event loop fetches requests from the waiting
        queue, forwards them to the context stage, and then puts them into the
        "bridge" queue.
      - Decoding stage event loop. This event loop accepts requests from the
        "bridge" queue (put them into the decoding queue), and then fetches requests
        from the decoding queue, forwards them to the decoding stage, and then
        informs the caller of the generated tokens.

    Note: Users may not use LLMEngine directly, but use more user-friendly wrapper classes
    OfflineLLM and AsyncLLM instead.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        context_sched_config: ContextStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig,
        context_devices: List[str] = None,
        decoding_devices: List[str] = None
    ):
        self.model_config = model_config
        self.disagg_parallel_config = disagg_parallel_config
        self.cache_config = cache_config
        self.context_sched_config = context_sched_config
        self.decoding_sched_config = decoding_sched_config

        self.request_counter = Counter()
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )
        
        self.bridge_queue = asyncio.Queue()

        if not ray.is_initialized():
            ray.init(
                # include_dashboard=False
                address="ray://219.222.20.79:30807"
            )
        
        logger.info("Initializing placement group")
        # placement_groups = self._init_placement_groups()
        
        if context_devices != None and decoding_devices != None:
            assert (
                len(context_devices) == disagg_parallel_config.context.pipeline_parallel_size * disagg_parallel_config.context.tensor_parallel_size
                and
                len(decoding_devices) == disagg_parallel_config.decoding.pipeline_parallel_size * disagg_parallel_config.decoding.tensor_parallel_size
            ), "num of available devices does not fits parallel_config"

        self.context_devices = context_devices
        self.decoding_devices = decoding_devices

        self.node_resources = {}
        self._init_inspect()

        self.device_map = {}
        self._device_nodeid_mapping()

        context_deployment, decoding_deployment = self._init_deployments()
        
        logger.info("Initializing context stage LLM engine")
        self.context_engine = ContextStageLLMEngine(
            self.bridge_queue,
            model_config,
            disagg_parallel_config.context,
            cache_config,
            context_sched_config,
            context_deployment,
            self._on_new_step_output_callback,
            self._on_new_lifetime_event_callback
        )
        
        logger.info("Initializing decoding stage LLM engine")
        self.decoding_engine = DecodingStageLLMEngine(
            self.bridge_queue,
            model_config,
            disagg_parallel_config.decoding,
            cache_config,
            decoding_sched_config,
            decoding_deployment,
            self.context_engine.clear_migrated_blocks_callback,
            self._on_new_step_output_callback,
            self._on_new_lifetime_event_callback,
            self.context_engine.workers
        )
        
        # request_id -> list of StepOutput
        # Created when calling self.generate()
        # Cleared when the request is finished
        self.request_outputs: Dict[int, asyncio.Queue[StepOutput]] = {}
        
        # request_id -> list of LifetimeEvent
        # Created when calling self.generate()
        # Cleared by the caller of self.generate() (i.e. the engine does not clear that)
        # TODO: clear this automatically to avoid memory leak
        self.request_lifetime_events: Dict[int, List[LifetimeEvent]] = {}
        
        self.engine_initialized = False

    def _init_inspect(self):
        nodes = ray.nodes()
        futures = []
        for i, node in enumerate(nodes):
            if node['Alive']:
                node_id = node['NodeID']
                future = resource_inspect.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote()
                futures.append((node_id, future))
        # Save & Print out Inspects Data
        for idx, (node_id, future) in enumerate(futures):
            result = ray.get(future)
            self.node_resources[node_id] = result
            # outlog = ''
            # outlog += f'[Worker{idx}] NodeID: {node_id}\n'
            # outlog += f'[Worker{idx}] Device: {self.device_map[node_id]}\n'
            # outlog += f'[Worker{idx}] Used/Total VRAM: {result["Used_VRAM"]/1024:.1f}/{result["Total_VRAM"]/1024:.1f} GB ({(result["Used_VRAM"]/result["Total_VRAM"])*100:.1f}%)\n'
            # outlog += f'[Worker{idx}] Free VRAM: {result["Free_VRAM"]/1024:.1f} GB\n'
            # print(outlog)

    def _device_nodeid_mapping(self):
        nodes = ray.nodes()
        for node in nodes:
            if node['Alive']:
                node_id = node['NodeID']
                resources = list(node['Resources'].keys())
                for el in resources:
                    if el.startswith('device'):
                        device = el.split(':',1)[1]
                self.device_map[node_id] = device

    def _init_deployments(self):
        selected = []

        def get_deployment(deployment:List, available_devices:List):
            if len(available_devices) > 0:
                device_map_rvt = {v:k for k,v in self.device_map.items()}
                for node in available_devices:
                    if node not in device_map_rvt:
                        raise RuntimeError(f"Node [{node}] not found in cluster")
                    if node in selected:
                        raise RuntimeError(f"Node [{node}] is already used")
                    deployment.append(device_map_rvt[node])
                    selected.extend([node, device_map_rvt[node]])
            else:
                for node_id, res in self.node_resources.items():
                    if res['Free_VRAM'] > 4096 and 'jetson' in self.device_map[node_id] and node_id not in selected:
                        deployment.append(node_id)
                        selected.extend([self.device_map[node_id], node_id])
            return deployment

        context_deployment = []
        decoding_deployment = []
        return get_deployment(context_deployment, self.context_devices), \
                get_deployment(decoding_deployment, self.decoding_devices)
    
    def _on_new_step_output_callback(self, request_id: int, step_output: StepOutput):
        """
        Called by self.context_engine or self.decoding_engine when a new output token
        is generated
        """
        self.request_outputs[request_id].put_nowait(step_output)
        
    def _on_new_lifetime_event_callback(self, request_id: int, event: LifetimeEvent, dont_add_if_dup: bool = False):
        """
        Called by self.context_engine or self.decoding_engine when a new lifetime event
        is generated
        """
        # if dont_add_if_dup == True and self.request_lifetime_events[request_id][-1].event_type == event.event_type, don't add it
        if dont_add_if_dup and \
            len(self.request_lifetime_events[request_id]) > 0 and \
                self.request_lifetime_events[request_id][-1].event_type == event.event_type:
            return
        self.request_lifetime_events[request_id].append(event)
    
    def _init_placement_groups(self) -> Optional[List[PlacementGroup]]:
        """
        Create placement groups for all engines and all workers
        
        Currently we force the same layer of the context & decoding stage to be executed
        on the same node (we call this "aligned"). This simplifies k/v cache migration.
        """
        context_pp = self.disagg_parallel_config.context.pipeline_parallel_size
        context_tp = self.disagg_parallel_config.context.tensor_parallel_size
        decoding_pp = self.disagg_parallel_config.decoding.pipeline_parallel_size
        decoding_tp = self.disagg_parallel_config.decoding.tensor_parallel_size
        
        # Each placement group is responsible for `layer_per_placement_group` layers
        layer_per_context_pp = self.model_config.get_num_layers(self.disagg_parallel_config.context)
        layer_per_decoding_pp = self.model_config.get_num_layers(self.disagg_parallel_config.decoding)
        layer_per_placement_group = math.lcm(layer_per_context_pp, layer_per_decoding_pp)
        
        # Each placement group contains `workers_per_placement_group` workers
        workers_per_placement_group = \
            layer_per_placement_group // layer_per_context_pp * context_tp \
            + layer_per_placement_group // layer_per_decoding_pp * decoding_tp
        
        # There should be `num_placement_groups` placement groups in total
        num_placement_groups = self.model_config.get_num_layers() // layer_per_placement_group
        assert num_placement_groups * workers_per_placement_group == \
            context_pp * context_tp + decoding_pp * decoding_tp
        
        # Create placement groups
        placement_groups = []
        for i in range(num_placement_groups):
            placement_group = ray.util.placement_group(
                [ { "GPU": 1 }] * workers_per_placement_group,
                strategy="STRICT_PACK",
            )
            ray.get(placement_group.ready(), timeout=1000)
            placement_groups.append(placement_group)
        
        return placement_groups
        
    async def initialize(self):
        await asyncio.gather(
            self.context_engine.initialize(),
            self.decoding_engine.initialize()
        )
        self.decoding_engine.init_migrate_pairs()
        self.engine_initialized = True
        
    def _remote_call_all_workers(
        self, 
        func_name: str, 
        *args
    ):
        """
        call func_name on all workers, blocked until all workers finish, and return all the results
        """
        handlers = self._remote_call_all_workers_async(func_name, *args)
        return ray.get(handlers)

    def _remote_call_all_workers_async(
        self, 
        func_name: str,
        *args
    ):
        """
        call func_name asynchronously on all workers (context/decoding/both), return the futures immediately
        """
        handlers = self.context_engine._remote_call_all_workers_async(func_name, *args)
        handlers += self.decoding_engine._remote_call_all_workers_async(func_name, *args)
        return handlers

    async def _start_my_event_loop(self):
        pass
    
    async def start_all_event_loops(self):
        """
        start_all_event_loops: Start context_engine's, decoding_engine's, and
        mine (LLMEngine's) event loops
        """
        logger.info("Starting LLMEngine's event loops")
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before starting event loops."
        await asyncio.gather(
            self.context_engine.start_event_loop(),
            self.decoding_engine.start_event_loop(),
            self._start_my_event_loop()
        )
        
    async def generate(
        self,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[str]],
        sampling_params: SamplingParams,
        arrival_time: Optional[float] = None,
        request_id: Optional[int] = None,
    ) -> AsyncGenerator[StepOutput, None]:
        """
        generate - Generate outputs for one request
        
        This function is intended to be used as an async generator, i.e., it can be
        used in a for loop. For example, `async for output in engine.generate(...)`
        """
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before generating."
        req = create_request(
            prompt,
            prompt_token_ids,
            sampling_params,
            self.request_counter,
            self.tokenizer,
            arrival_time,
            request_id,
        )
        self.request_outputs[req.request_id] = asyncio.Queue()
        self.request_lifetime_events[req.request_id] = []
        
        self._on_new_lifetime_event_callback(req.request_id, LifetimeEvent(LifetimeEventType.Issued))
        self.context_engine.add_request(req)
        
        while True:
            try:
                step_output = await self.request_outputs[req.request_id].get()
            except asyncio.CancelledError:
                # The engine returns
                # Exception should be handled by the engine, not me
                return
            except GeneratorExit:
                return
            yield step_output
            if step_output.is_finished:
                break
                
        del self.request_outputs[req.request_id]

    def abort_request(self, request_id: int):
        self.context_engine.abort_request(request_id)
        self.decoding_engine.abort_request(request_id)
        
def add_engine_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-dummy-weights", action="store_true")
    
    parser.add_argument("--context-pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--context-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--decoding-tensor-parallel-size", type=int, default=1)
    
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-blocks-per-req", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=int, default=16)
    
    parser.add_argument("--context-sched-policy", type=str, default="fcfs")
    parser.add_argument("--context-max-batch-size", type=int, default=256)
    parser.add_argument("--context-max-tokens-per-batch", type=int, default=4096)
    
    parser.add_argument("--decoding-sched-policy", type=str, default="fcfs")
    parser.add_argument("--decoding-max-batch-size", type=int, default=256)
    parser.add_argument("--decoding-max-tokens-per-batch", type=int, default=8192)
    
    parser.add_argument("--simulator-mode", action="store_true")
    parser.add_argument("--profiler-data-path", type=str, default=None)
    parser.add_argument("--gpu-mem-size-gb", type=float, default=None)
    