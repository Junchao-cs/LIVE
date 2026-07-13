import os
import re
import random
import fsspec
import functools
import importlib
import contextlib
import subprocess
import numpy as np
from functools import partial
from inspect import isfunction
from rich import print
from typing import Union
from collections import namedtuple

import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file as load_safetensors

import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def instantiate_from_config_kwargs(config, **kwargs):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("[sgm.util][instantiate_from_config_kwargs] Expected key `target` to instantiate.")
    params = dict(config.get("params", dict()))
    params.update(kwargs)
    return get_obj_from_str(config["target"])(**params)


def get_valid_dirs(dir1: str, dir2: str, dir3: Union[None, str] = None) -> Union[None, str]:
    if (dir1 is not None) and os.path.isdir(dir1): return dir1
    elif (dir2 is not None) and os.path.isdir(dir2): return dir2
    elif (dir3 is not None) and os.path.isdir(dir3): return dir3
    else: return None


def get_valid_paths(path1: str, path2: str, path3: Union[None, str] = None) -> Union[None, str]:
    if (path1 is not None) and os.path.isfile(path1): return path1
    elif (path2 is not None) and os.path.isfile(path2): return path2
    elif (path3 is not None) and os.path.isfile(path3): return path3
    else: return None


def get_valid_dirs_or_paths(dir_or_path1: str, dir_or_path2: str, dir_or_path3: Union[None, str] = None) -> Union[None, str]:
    if (dir_or_path1 is not None) and (os.path.isdir(dir_or_path1) or os.path.isfile(dir_or_path1)): return dir_or_path1
    elif (dir_or_path2 is not None) and (os.path.isdir(dir_or_path2) or os.path.isfile(dir_or_path2)): return dir_or_path2
    elif (dir_or_path3 is not None) and (os.path.isdir(dir_or_path3) or os.path.isfile(dir_or_path3)): return dir_or_path3
    else: return None


def get_valid_dirs_idx(dir1: str, dir2: str, dir3: Union[None, str] = None) -> Union[None, str]:
    if (dir1 is not None) and os.path.isdir(dir1): return 0
    elif (dir2 is not None) and os.path.isdir(dir2): return 1
    elif (dir3 is not None) and os.path.isdir(dir3): return 2
    else: return None


def disabled_train_func(models: list):
    for model in models:
        model.eval()
        model.train = disabled_train
        for p in model.parameters():
            p.requires_grad = False


def seed_anything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_anytorch(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@rank_zero_only
def print0(*args, **kwargs):
    print(*args, **kwargs)  # python -m rich.color


def check_nvlink_connectivity(fabric=None):
    """Checks GPU connectivity for both NVIDIA and AMD GPUs.

    This function delegates to vendor-specific implementations based on
    the detected GPU vendor.
    """
    if fabric is not None:
        custom_print = fabric.print
    else:
        custom_print = print0

    if os.getenv("RANK", "0") == "0":
        try:
            if torch.cuda.is_available():
                device_properties = torch.cuda.get_device_properties(0)
                gpu_name = device_properties.name.lower()
                if "nvidia" in gpu_name:
                    _check_nvidia_connectivity(custom_print)
                elif "advanced micro devices" in gpu_name or "amd" in gpu_name:
                    _check_amd_connectivity(custom_print)
                else:
                    custom_print(f"[bold red]\[sgm.util][check_nvlink_connectivity][/bold red] Unrecognized GPU vendor: {device_properties.name}")
            else:
                custom_print("[bold red]\[sgm.util][check_nvlink_connectivity][/bold red] No GPUs available")
        except Exception as e:
            custom_print(f"[bold red]\[sgm.util][check_nvlink_connectivity][/bold red] An error occurred while checking GPU connectivity: {e}")


def _check_nvidia_connectivity(custom_print):
    """Checks NVLink connectivity on NVIDIA GPUs."""
    result = subprocess.run(["nvidia-smi", "topo", "-m"], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        custom_print("[bold red]\[sgm.util][_check_nvidia_connectivity][/bold red] Failed to run nvidia-smi")
        return

    lines = result.stdout.strip().split("\n")
    start_index = next((i for i, line in enumerate(lines) if "GPU0" in line), None)
    if start_index is None:
        custom_print("[bold red]\[sgm.util][_check_nvidia_connectivity][/bold red] Failed to parse nvidia-smi output")
        return

    headers_line = lines[start_index]
    headers = headers_line.split()
    gpu_regex = re.compile(r"^GPU\d+$")
    gpu_count = len([header for header in headers if gpu_regex.match(header)])

    all_nvlink = True
    for line in lines[start_index + 1 : start_index + 1 + gpu_count]:
        columns = line.split()
        connections = columns[1 : 1 + gpu_count]
        if not all("NV" in conn for conn in connections if conn != "X"):
            all_nvlink = False
            break

    if all_nvlink:
        custom_print("[bold red]\[sgm.util][_check_nvidia_connectivity][/bold red] All GPUs are fully connected via NVLink.")
    else:
        custom_print(
            "[bold red]\[sgm.util][_check_nvidia_connectivity][/bold red] Warning: Not all GPUs are fully connected via NVLink. Some GPUs are connected via slower interfaces. "
            "It is recommended to switch to a different machine with faster GPU connections for optimal multi-GPU training performance."
        )


def _check_amd_connectivity(custom_print):
    """Checks XGMI connectivity on AMD GPUs."""
    result = subprocess.run(["rocm-smi", "--showtopotype"], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        custom_print("[bold red]\[sgm.util][_check_amd_connectivity][/bold red] Failed to run rocm-smi")
        return

    lines = result.stdout.strip().split("\n")
    gpu_header_index = next((i for i, line in enumerate(lines) if re.match(r"^\s*GPU0", line)), None)
    if gpu_header_index is None or gpu_header_index == 0:
        custom_print("[bold red]\[sgm.util][_check_amd_connectivity][/bold red] Failed to parse rocm-smi output (no GPU headers found)")
        return

    header_line = lines[gpu_header_index - 1]
    headers = header_line.strip().split()
    gpu_regex = re.compile(r"^GPU\d+$")
    gpu_count = len([header for header in headers if gpu_regex.match(header)])

    gpu_lines = []
    for line in lines[gpu_header_index : gpu_header_index + gpu_count]:
        if re.match(r"^\s*GPU\d+", line):
            gpu_lines.append(line.strip())
    if len(gpu_lines) != gpu_count:
        custom_print("[bold red]\[sgm.util][_check_amd_connectivity][/bold red] Mismatch in GPU count when parsing rocm-smi output")
        return

    all_xgmi = True
    for line in gpu_lines:
        columns = line.split()
        connections = columns[1 : 1 + gpu_count]
        for conn in connections:
            if conn not in ("XGMI", "0"):
                all_xgmi = False
                break
        if not all_xgmi:
            break

    if all_xgmi:
        custom_print("[bold red]\[sgm.util][_check_amd_connectivity][/bold red] All GPUs are fully connected via XGMI.")
    else:
        custom_print(
            "[bold red]\[sgm.util][_check_amd_connectivity][/bold red] Warning: Not all GPUs are fully connected via XGMI. Some GPUs are connected via slower interfaces. "
            "It is recommended to switch to a different machine with faster GPU connections for optimal multi-GPU training performance."
        )


GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
        "power_draw",
    ],
)


class GPUMemoryMonitor:
    """
    Class to monitor GPU memory usage
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)  # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        # reset stats, clear cache
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = cuda_info["num_alloc_retries"]
        num_ooms = cuda_info["num_ooms"]
        power_draw = torch.cuda.power_draw()

        if num_retries > 0:
            print(f"[bold red]\[sgm.util][GPUMemoryMonitor][/bold red] {num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            print(f"[bold red]\[sgm.util][GPUMemoryMonitor][/bold red] {num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
            power_draw,
        )

    def reset_peak_stats(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def __str__(self):
        mem_stats = self.get_peak_stats()
        display_str = f"{self.device_name} ({self.device_index}): {self.device_capacity_gib} GiB capacity, "
        display_str += (
            f"{mem_stats.max_reserved_gib} GiB peak, {mem_stats.max_reserved_pct}% peak"
        )
        return f"{display_str}"


def report_gpu_memory(rank: Union[int, str] = 0, prefix: str = "sgm.util"):
    if isinstance(rank, str):
        rank = int(rank)
    # GPU check
    torch.cuda.synchronize()
    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    gpu_memory_monitor = GPUMemoryMonitor(f"cuda:{rank}")
    print(
        f"[bold red]\[{prefix}][/bold red] Rank {rank}: "
        f"{gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}): "
        f"Free / Total GPU memory: {free_gpu_memory/1024**3:.2f} / {total_gpu_memory/1024**3:.2f} GB"
    )


@contextlib.contextmanager
def clean_env():
    distrib_names = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "DORA_FORCE_DISTRIB",
    )
    cluster_env = {
        x: os.environ.pop(x)
        for x in os.environ
        if x.startswith(
            ("SLURM_", "SLURMD_", "SRUN_", "SBATCH_", "SUBMITIT_", "WANDB_")
        )
        or x in distrib_names
    }
    try:
        yield
    finally:
        os.environ.update(cluster_env)


#################################################################################
#                               generative-models                               #
#################################################################################


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def get_string_from_tuple(s):
    try:
        # Check if the string starts and ends with parentheses
        if s[0] == "(" and s[-1] == ")":
            # Convert the string to a tuple
            t = eval(s)
            # Check if the type of t is tuple
            if type(t) == tuple:
                return t[0]
            else:
                pass
    except:
        pass
    return s


def is_power_of_two(n):
    """
    chat.openai.com/chat
    Return True if n is a power of 2, otherwise return False.

    The function is_power_of_two takes an integer n as input and returns True if n is a power of 2, otherwise it returns False.
    The function works by first checking if n is less than or equal to 0. If n is less than or equal to 0, it can't be a power of 2, so the function returns False.
    If n is greater than 0, the function checks whether n is a power of 2 by using a bitwise AND operation between n and n-1. If n is a power of 2, then it will have only one bit set to 1 in its binary representation. When we subtract 1 from a power of 2, all the bits to the right of that bit become 1, and the bit itself becomes 0. So, when we perform a bitwise AND between n and n-1, we get 0 if n is a power of 2, and a non-zero value otherwise.
    Thus, if the result of the bitwise AND operation is 0, then n is a power of 2 and the function returns True. Otherwise, the function returns False.

    """
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast


def load_partial_from_config(config):
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


def log_txt_as_img(wh, xc, size=10, split_loc=40):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(split_loc * (wh[0] / 256))
        if isinstance(xc[bi], list):
            text_seq = xc[bi][0]
        else:
            text_seq = xc[bi]
        lines = "\n".join(
            text_seq[start : start + nc] for start in range(0, len(text_seq), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("[sgm.util][log_txt_as_img] Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def make_path_absolute(path):
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return os.path.abspath(p)
    return path


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def isheatmap(x):
    if not isinstance(x, torch.Tensor):
        return False

    return x.ndim == 2


def isneighbors(x):
    if not isinstance(x, torch.Tensor):
        return False
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


def exists(x):
    return x is not None


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("[sgm.util][instantiate_from_config] Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"[sgm.util][append_dims] input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def load_model_from_config(config, ckpt, verbose=True, freeze=True):
    print(f"Loading model from {ckpt}")
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"[sgm.util][load_model_from_config] Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError

    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[sgm.util][load_model_from_config] missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("[sgm.util][load_model_from_config] unexpected keys:")
        print(u)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.eval()
    return model


def get_configs_path() -> str:
    """
    Get the `configs` directory.
    For a working copy, this is the one in the root of the repository,
    but for an installed copy, it's in the `sgm` package (see pyproject.toml).
    """
    this_dir = os.path.dirname(__file__)
    candidates = (
        os.path.join(this_dir, "configs"),
        os.path.join(this_dir, "..", "configs"),
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"[sgm.util][get_configs_path] Could not find SGM configs in {candidates}")


def get_nested_attribute(obj, attribute_path, depth=None, return_key=False):
    """
    Will return the result of a recursive get attribute call.
    E.g.:
        a.b.c
        = getattr(getattr(a, "b"), "c")
        = get_nested_attribute(a, "b.c")
    If any part of the attribute call is an integer x with current obj a, will
    try to call a[x] instead of a.x first.
    """
    attributes = attribute_path.split(".")
    if depth is not None and depth > 0:
        attributes = attributes[:depth]
    assert len(attributes) > 0, "At least one attribute should be selected"
    current_attribute = obj
    current_key = None
    for level, attribute in enumerate(attributes):
        current_key = ".".join(attributes[: level + 1])
        try:
            id_ = int(attribute)
            current_attribute = current_attribute[id_]
        except ValueError:
            current_attribute = getattr(current_attribute, attribute)

    return (current_attribute, current_key) if return_key else current_attribute
