import torch
import random
import math
import numpy as np
from itertools import islice

a = torch.arange(0, 1000, 1)
density = lambda t: (1 - 0.5 * math.cos(math.pi * t / 1000)) / 1000
aa = torch.tensor([density(aaa) for aaa in a])
timesteps = torch.multinomial(aa, 8, replacement=True)
print(timesteps.dtype)
timesteps_old = torch.randint(0, 1000, (8,))
print(timesteps_old.dtype)
