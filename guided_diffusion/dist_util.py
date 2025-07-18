"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(args):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    if not args.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "172.17.0.2"
    else:
        # hostname = socket.gethostbyname(socket.getfqdn())
        hostname = "172.17.0.2"
    os.environ["MASTER_ADDR"] = hostname #comm.bcast(hostname, root=0)
    os.environ["RANK"] = '0'#str(comm.rank)
    os.environ["WORLD_SIZE"] = '1'#str(comm.size)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     mpigetrank=0
#     if mpigetrank==0:
#         with bf.BlobFile(path, "rb") as f:
#             data = f.read()
#     else:
#         data = None
    
#     return th.load(io.BytesIO(data), **kwargs)
def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file using the standard PyTorch method.
    """
    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
