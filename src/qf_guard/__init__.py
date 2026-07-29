import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

__all__ = ["__version__"]
__version__ = "0.1.0"
