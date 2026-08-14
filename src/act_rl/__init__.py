"""Loosely coupled offline-to-online reinforcement learning for LeRobot ACT.

The core package deliberately does not import LeRobot or the Marvin SDK.  Those
dependencies live behind adapters so replay, critics and IQL can be tested on CPU.
"""

from .config import ACTRLConfig, IQLConfig, NetworkConfig
from .types import ChunkBatch, ChunkTransition

__all__ = ["ACTRLConfig", "IQLConfig", "NetworkConfig", "ChunkBatch", "ChunkTransition"]
__version__ = "0.1.0"
