"""Small, local Code-RL building blocks.

The reward checker is deliberately independent from the optional AReaL
trainer used by the upstream ZGCM-1 repository.
"""

from .code_reward import CodeReward, RewardResult

__all__ = ["CodeReward", "RewardResult"]
