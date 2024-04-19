import sys
import os

sys.path.append(os.path.join(sys.path[0], '..'))

from typing import Dict

import numpy as np
from bayes_net import Node, BayesNetwork
from distributions import Prior, CPT



# create nodes
names_list = ["primo", "secondo", "terzo", "quarto", "quinto", "sesto"]

nodes_list = [Node(label=name, node_id=i) for i, name in enumerate(names_list,1)]


arcs_list = []
for i in range(1,6):
    arcs_list.append((i,i+1))

BN = BayesNetwork(nodes_list,arcs_list)


def build_dummy_cpt(parent_n:str, 
                    p0:float=0.5, 
                    p1:float=0.5) -> Dict[frozenset,float]:
    cpt = {
        frozenset([(parent_n,0)]) : p0,
        frozenset([(parent_n,1)]) : p1
    }
    return cpt

nodes_list[0].assign_CPT(p=0.5)
for i, nome in enumerate(names_list[:-1],1):
    node_cpt = build_dummy_cpt(nome,p0:=0.5,p1:=0.5)
    nodes_list[i].assign_CPT(node_cpt)

# check that initially all nodes don't have latest samples
for n in nodes_list:
    assert n.distribution._latest_sample is None 

s = 0
for _ in range(10000):
    s += nodes_list[-1].distribution.sample_under_evidence([('quarto',1)])
s = s/10000

# these assertions basically check that we haven't
# sampled for nodes earlier than "quarto" (shouldn't have been done...)
for n in nodes_list[:4]:
    assert n.distribution._latest_sample is None

# we assume p0=p1 (so the distribution of each node in practice is 
# independent from that of the parent), in particular p0=p1=0.5
assert s < p0 + 0.01 and s > p0 - 0.01

#s = 0
#for _ in range(10000):
#    s += nodes_list[-1].distribution.sample()
#
#s = s/10000