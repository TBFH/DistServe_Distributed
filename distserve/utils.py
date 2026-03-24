import numpy as np
import psutil
import random
import subprocess as sp
import torch
import uuid
from typing import TypeAlias, List
from enum import Enum
import requests

GB = 1 << 30
MB = 1 << 20


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_gpu_memory_usage(gpu: int = 0):
    """
    Python equivalent of nvidia-smi, copied from https://stackoverflow.com/a/67722676
    and verified as being equivalent ✅
    """
    output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"

    try:
        memory_use_info = output_to_list(
            sp.check_output(COMMAND.split(), stderr=sp.STDOUT)
        )[1:]

    except sp.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )

    return int(memory_use_info[gpu].split()[0])


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


cudaMemoryIpcHandle: TypeAlias = List[int]


class Stage(Enum):
    """The stage of a SingleStageLLMEngine"""

    CONTEXT = "context"
    DECODING = "decoding"

    def __str__(self) -> str:
        return self.value



def power_plot(devices, power_data, save_path):
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    data = power_data

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 四个不同的颜色
    # colors = plt.cm.tab20(np.linspace(0, 1, 20))
    # colors = plt.cm.Set3(np.linspace(0, 1, 16))

    # 2. 定义16种线型和标记组合
    line_styles = ['-', '--', '-.', ':'] * 4  # 重复使用以确保足够
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_', '1']  # 16种不同的标记

    # 创建图形
    plt.figure(figsize=(8, 5))

    # 3. 控制图例显示顺序（这里按设备名称排序，您可以自定义顺序）
    # 例如：按设备内存大小排序
    legend_order = {device:idx for idx, device in enumerate(devices)}

    # 按指定顺序排序设备
    sorted_devices = sorted(data.items(), key=lambda x: legend_order.get(x[0], 99))

    # 收集所有功耗值用于确定纵轴范围
    all_power_values = []

    # 绘制每条曲线
    for idx, (device_name, device_data) in enumerate(sorted_devices):
        # 提取时间戳和功耗值
        timestamps = [item[0] for item in device_data]
        power_values = [item[1] for item in device_data]
        
        # 收集所有功耗值
        all_power_values.extend(power_values)
        
        # 从0开始的时间（以第一个时间戳为基准）
        start_time = min(timestamps)
        relative_times = [ts - start_time for ts in timestamps]
        
        # 绘制曲线
        plt.plot(relative_times, power_values, 
                color=colors[idx % len(colors)],
                linestyle=line_styles[0 % len(line_styles)],
                marker=markers[0 % len(markers)],
                linewidth=2,
                markersize=4,
                label=device_name)

    # 设置图形属性
    plt.title('Power Curve', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Power (W)', fontsize=14)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. 自适应纵轴显示范围（基于数据范围，上下留10%的边距）
    if all_power_values:
        min_power = min(all_power_values)
        max_power = max(all_power_values)
        power_range = max_power - min_power
        
        # 计算纵轴范围，上下留出10%的边距
        y_margin = power_range * 0.1
        y_lower = max(0, min_power - y_margin)  # 确保不低于0
        y_upper = max_power + y_margin
        
        plt.ylim(y_lower, y_upper)
        
        # 在纵轴上添加一条参考线（0功率线，如果需要）
        if y_lower <= 0:
            plt.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # 设置x轴范围（从0开始，稍微留点边距）
    all_timestamps = [item[0] for sublist in data.values() for item in sublist]
    if all_timestamps:
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        time_range = end_time - start_time
        plt.xlim(-time_range*0.05, time_range*1.05)  # 左边留5%，右边留5%

    # 添加图例
    plt.legend(fontsize=10, loc='best', framealpha=0.95, ncol=2)  # ncol=2可以让图例分两列显示

    # 添加数值标签（可选）
    # for idx, (device_name, device_data) in enumerate(sorted_devices):
    #     timestamps = [item[0] for item in device_data]
    #     power_values = [item[1] for item in device_data]
    #     start_time = min(timestamps)
    #     relative_times = [ts - start_time for ts in timestamps]
        
    #     # 在每个数据点上添加数值标签
    #     for x, y in zip(relative_times, power_values):
    #         plt.text(x, y, f'{y}', fontsize=9, ha='center', va='bottom')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    # plt.show()

    # 保存图形（可选）
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_path, f"power-plot_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"power plot saved at {save_path}")

def sort_power_data(*fetched_list):
    res = {}
    for fetched in fetched_list:
        if fetched['status'] != 'success':
            print('Fetched Not Success')
            return None
        for node in fetched['data']['result']:
            node_name = node['metric']['instance']
            if 'jetson' in node_name:
                val = [[int(time), float(power)/1000] for time, power in node['values']]
                res[node_name] = val
            elif 'pc' in node_name:
                val = [[int(time), float(power)] for time, power in node['values']]
                res[f"{node_name}-{node['metric']['gpu']}"] = val
    
    return res

def grafana_query_range(query, start, end, step):
    '''
    直接调用Grafana API接口查找过去一段时间内的所有监控数据，根据传入的PromQL语法字符串查找
    '''
    import os
    grafana_key = os.environ.get('GRAFANA_API_KEY')
    base_url = 'http://219.222.20.79:32411/api/datasources/proxy/1/api/v1/query_range'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Bearer {grafana_key}'
    }
    data = {
        "query": query,
        "type": 1,
        "start": start,
        "end": end,
        "step": step
    }
    # 发送http请求
    response = requests.post(
        base_url,
        headers=headers,
        data=data,
        timeout=30
    )
    # 处理响应
    if response.status_code == 200:
        res = response.json()
        return res
    elif response.status_code == 404:
        print("资源未找到")
        return None
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")

def get_avg_power(power_data, avg):
    avgs = {}
    for device, data in power_data.items():
        powers = [float(power) for _, power in data]
        if avg:
            avgs[device] = sum(powers)/len(powers)
        else:
            avgs[device] = powers
    return avgs

def profile_powers(devices, start, end, step, plot_dir=None, avg=True):
    devices_jetson=[]
    devices_pc=[]
    for d in devices:
        if 'jetson' in d:
            devices_jetson.append(d)
        elif 'pc' in d:
            devices_pc.append(d)
        else:
            raise ValueError(f"Power Profile Error: Unknown Type of Device {d}")
    # 接口调用参数
    jetson_query = f'integrated_power_mW{"{"}instance=~"({"|".join(devices_jetson)})", statistic="power"{"}"}'
    pc_query = f'DCGM_FI_DEV_POWER_USAGE{"{"}instance=~"({"|".join(devices_pc)})", gpu=~"(0|1)"{"}"}'
    end = int(end)
    start = int(start)
    if end - start >= 30:
        start -= 10
    step = step    # 表示每多少秒获取一次数据
    # 获取功耗数据
    jetson_fetched = grafana_query_range(jetson_query, start, end, step)
    pc_fetched = grafana_query_range(pc_query, start, end, step)
    # 格式化功耗数据
    power_data = sort_power_data(jetson_fetched, pc_fetched)
    # 保存功耗曲线图
    if plot_dir:
        power_plot(devices, power_data, plot_dir)
    # 计算平均功耗
    power_data = get_avg_power(power_data, avg)
    
    return power_data