import math
import random

from typing import Callable, Union

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torch import Tensor
from torch.distributions import Uniform
from torch.utils.data import WeightedRandomSampler

from rl4co.utils.ops import gather_by_index
from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class OBPGenerator(Generator):
    def __init__(self):
        pass

    def _generate(self, batch_size, **kwargs) -> TensorDict:
        pass
