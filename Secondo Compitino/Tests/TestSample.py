import io
import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
from bayes_net import Node, BayesNetwork
from distributions import PriorBernoulli, CPT

#create nodes
n1 = Node(label="primo", node_id = 1)
n2 = Node(label="secondo", node_id = 2)

#create bayes net
BN1 = BayesNetwork()

#add nodes to the net
BN1.add_nodes(n1)
BN1.add_nodes(n2)

# create arc and add it to the network
arc1 = (1,2)
BN1.add_arcs(arc1)

init_dict = {
    frozenset([('primo',0)]) : 0.1,
    frozenset([('primo',1)]) : 0.99
}
CPTprova = CPT(init_dict, n2.BS)
n1.distribution = PriorBernoulli(0.001)

CPTprova.sample()

s = 0 
for _ in range(10000):
    s = s + CPTprova.sample()

s = s / 10000
# very rough
assert s > 0.09 and s < 0.11