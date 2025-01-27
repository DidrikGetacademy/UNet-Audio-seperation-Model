import torch
import gc
import os
import sys
import psutil
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
from Training.Externals.utils import Return_root_dir



root_dir = Return_root_dir() #Gets the root directory
Memory_logger = setup_logger('Memory_debugging',os.path.join(root_dir,"Model_Performance_logg/log/Memory.txt"))


def log_memory_usage(tag=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Memory_logger.info(f"[Memory {tag}] Timestamp: {timestamp}")
    print(f"[Memory {tag}] Timestamp: {timestamp}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                device_name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9  # GB
                allocated_mem = torch.cuda.memory_allocated(i) / 1e9  # GB
                cached_mem = torch.cuda.memory_reserved(i) / 1e9  # GB

                memory_info = (
                    f"[Memory {tag}] GPU {i}: {device_name}, "
                    f"Total={total_mem:.2f} GB, Allocated={allocated_mem:.2f} GB, Cached={cached_mem:.2f} GB"
                )
                Memory_logger.info(memory_info)
                print(memory_info)
            except Exception as e:
                error_msg = f"[Memory {tag}] Error retrieving GPU memory for device {i}: {str(e)}"
                Memory_logger.error(error_msg)
                print(error_msg)
    else:
        info_msg = f"[Memory {tag}] CUDA is not available. Using CPU only."
        Memory_logger.info(info_msg)
        print(info_msg)

    process = psutil.Process(os.getpid())
    cpu_mem_psutil = process.memory_info().rss / 1e9  # GB
    cpu_memory_info = f"[Memory {tag}] CPU Memory Usage: ~{cpu_mem_psutil:.2f} GB"
    Memory_logger.info(cpu_memory_info)
    print(cpu_memory_info)





def clear_memory_before_training():
    log_memory_usage(tag="Clearing memory before training")
    torch.cuda.empty_cache()
    gc.collect()
    log_memory_usage(tag="Memory after clearing it...")



def log_memory_after_index_epoch(epoch):
    if epoch == 0 or (epoch + 1) % 5 == 0:  
                log_memory_usage(tag=f"After Epoch {epoch + 1}")

