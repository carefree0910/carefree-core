import os
import psutil

from typing import Dict
from typing import List
from typing import Optional
from collections import defaultdict


def format_num_bytes(num_bytes: float) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if num_bytes < 1024:
            break
        num_bytes /= 1024.0
    return f"{num_bytes:.2f}{unit}B"


def get_memory_info(pid: int) -> Dict[str, int]:
    res: Dict[str, int] = defaultdict(int)
    for mmap in psutil.Process(pid).memory_maps():
        res["rss"] += getattr(mmap, "rss", 0)
        res["pss"] += getattr(mmap, "pss", 0)
        private_clean = getattr(mmap, "private_clean", 0)
        private_dirty = getattr(mmap, "private_dirty", 0)
        res["uss"] += private_clean + private_dirty
        shared_clean = getattr(mmap, "shared_clean", 0)
        shared_dirty = getattr(mmap, "shared_dirty", 0)
        res["shared"] += shared_clean + shared_dirty
        if mmap.path.startswith("/"):
            res["shared_file"] += shared_clean + shared_dirty
    return res


class MemoryMonitor:
    def __init__(self, pids: Optional[List[int]] = None):
        self.pids = pids or [os.getpid()]

    def table(self, col_space: int = 10) -> str:
        import pandas as pd

        data = {pid: get_memory_info(pid) for pid in self.pids}
        keys = list(list(data.values())[0].keys())
        table = []
        for pid, pid_data in data.items():
            table.append([str(pid)] + [format_num_bytes(pid_data[k]) for k in keys])
        df = pd.DataFrame(table, columns=["PID"] + keys)
        return df.to_string(index=False, col_space=col_space)


__all__ = [
    "MemoryMonitor",
    "format_num_bytes",
    "get_memory_info",
]
