import numpy as np


class Bernoulli():
    def __init__(self, p:float):
        self._p = p

    def set_p(self, new_p:float) -> None:
        self._p = new_p

    def get_p(self) -> float:
        return self._p
    
    def sample(self) -> int:
        return int(np.random.random() < self._p)