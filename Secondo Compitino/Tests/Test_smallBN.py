import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from bayes_net import Node, BayesNetwork
from distributions import Prior, CPT

# create nodes
n1 = Node(label="prior1", node_id=1)
n2 = Node(label="prior2", node_id=2)
n3 = Node(label="terzo", node_id=3, distribution=Prior(0.2))

a1 = (1,3)
a2 = (2,3)

BN = BayesNetwork([n1,n2,n3], [a1,a2])

n1.assign_CPT(p=0.5)
n2.assign_CPT(p=0.99)

# specify the conditional distrib of a categorical RV
cpt_spec = {
    frozenset([('prior1',0),('prior2',0)]) : [0.2, 0.6, 0.2],
    frozenset([('prior1',0),('prior2',1)]) : [0.7, 0.05, 0.25],
    frozenset([('prior1',1),('prior2',0)]) : [0.2, 0.2, 0.6],
    frozenset([('prior1',1),('prior2',1)]) : [0.5, 0.05, 0.45]
}

n3.assign_CPT(full_cpt=cpt_spec)

s = np.zeros(3)
for _ in range(10000):
    idx = n3.distribution.sample()
    s[idx-1] += 1

s = s / 10000

# s ~ [0.6, 0.05, 0.35]
assert s[0] < 0.615 and s[0] > 0.585
assert s[1] < 0.065 and s[1] > 0.035
assert s[2] < 0.455 and s[2] > 0.245

# test that the CPTs can be reassigned
nprova = Node(label='prova',node_id=42,distribution=Prior(0.6))
nprova.assign_CPT(p=0.55)

assert nprova.distribution.get_pvec() == 0.55