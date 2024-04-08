from typing import List, Tuple, Union

Arc = Tuple[int,int]    # datatype for arc in the Bayes network

P = Union[float, List[float]]    # probabilities in the CPT
PassedConditions = frozenset[Tuple[str,int]]    # rows of the CPT (the passed ones)