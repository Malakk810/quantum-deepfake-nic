import os
import random
import socket
import platform

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def machine_name():
    # Kaggle and Colab hostnames are random, so set MACHINE_NAME yourself
    return os.environ.get("MACHINE_NAME", socket.gethostname())


def env_info():
    info = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ["torch", "sklearn", "pennylane"]:
        try:
            module = __import__(name)
            info[name] = module.__version__
        except ImportError:
            info[name] = "missing"
    return info
