import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
from bayes_net import Node, BayesNetwork
from distributions import Prior, CPT

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
CPTprova = CPT(init_dict, parents_list=n2.BS, node_label=n2.label)
n1.distribution = Prior(0.001)
#n1.assign_CPT(0.001)    


CPTprova.sample()

s = 0 
for _ in range(10000):
    s = s + CPTprova.sample()

s = s / 10000
# very rough
assert s > 0.09 and s < 0.11



# wrong assignment:
n3 = Node(label='terzo')
BN1.add_nodes(n3)
BN1.add_arcs((3,2))

wrong_init_dict = {
    frozenset([('primo',0),('terzo',0)]) : 0.1,
    frozenset([('primo',1)]) : 0.99
}
CPTwrong = CPT(wrong_init_dict, n2.BS, n2.label)