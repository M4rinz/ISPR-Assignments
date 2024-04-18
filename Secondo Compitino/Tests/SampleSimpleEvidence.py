import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from bayes_net import Node, BayesNetwork
from distributions import Prior, CPT

# create nodes
n1 = Node(label="prior1", node_id=1)
n2 = Node(label="prior2", node_id=2)
n3 = Node(label="terzo", node_id=3)

a1 = (1,3)
a2 = (2,3)

BN = BayesNetwork([n1,n2,n3], [a1,a2])

n1.assign_CPT(p=0.1)
n2.assign_CPT(p=0.99)

# specify the conditional distrib (I'll think about something smart)
cpt_spec = {
    frozenset([('prior1',0),('prior2',0)]) : 0.99,
    frozenset([('prior1',0),('prior2',1)]) : 0.99,
    frozenset([('prior1',1),('prior2',0)]) : 0.5,
    frozenset([('prior1',1),('prior2',1)]) : 0.5
}

n3.assign_CPT(full_cpt=cpt_spec)

evidenza = [('prior1',1)]