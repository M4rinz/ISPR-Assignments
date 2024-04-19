import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from bayes_net import Node, BayesNetwork
from distributions import Prior, CPT

# create nodes
n1 = Node(label="primo", node_id=1)
n2 = Node(label="secondo", node_id=2)
n3 = Node(label="terzo", node_id=3)
n4 = Node(label="quarto", node_id=4)
n5 = Node(label="quinto", node_id=5)
n6 = Node(label="sesto", node_id=6)

arcs_list = []
for i in range(1,6):
    arcs_list.append((i,i+1))

BN = BayesNetwork([n1,n2,n3,n4,n5,n6],arcs_list)