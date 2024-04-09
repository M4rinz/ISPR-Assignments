import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
from bayes_net import Node, BayesNetwork

import numpy as np


# create prior nodes
n1 = Node(label="OkSoprintendenza", node_id = 1)
n2 = Node(label="sound_check", node_id = 2)
n3 = Node(label="OkWeather", node_id = 3)
n4 = Node(label="OkDrummer", node_id = 4)

# create inner nodes
n5 = Node(label="OkBureaucracy", node_id=5)
n6 = Node(label="concert_held", node_id=6)

# create arcs
a1 = (1,5)
a2 = (2,5)
a3 = (3,6)
a4 = (4,6)
a5 = (5,6)

# create bayes net
BN = BayesNetwork([n1,n2,n3,n4,n5,n6], [a1,a2,a3,a4,a5])

# create distributions
n1.assign_CPT(p = 0.85)     # la soprintendenza può non dare l'ok con una probabilità del 15%
n2.assign_CPT(p = [0.8, 0.15, 0.05])       # sound threshold check can be 1 = ok, 2 = too loud, 3 = far too loud
n3.assign_CPT(p = 0.9)      # weather has a 10% chance of being bad
n4.assign_CPT(p = 0.8)       # the drummer can be sick with a 20% chance

# create the CPT for the bureaucracy
bureaucracy_cpt = {
    frozenset([('soprintendenza',0),('sound_check',3)]) : 0.05,
    frozenset([('soprintendenza',0),('sound_check',2)]) : 0.10,
    frozenset([('soprintendenza',0),('sound_check',1)]) : 0.25,
    frozenset([('soprintendenza',1),('sound_check',3)]) : 0.40,
    frozenset([('soprintendenza',1),('sound_check',2)]) : 0.75,
    frozenset([('soprintendenza',1),('sound_check',1)]) : 0.95
}
n5.assign_CPT(full_cpt=bureaucracy_cpt)

# create the CPT for the concert
concert_cpt = {
    frozenset([('OkBureaucracy',0),('OkWeather',0),('OkDrummer',0)]) : 0.02,
    frozenset([('OkBureaucracy',0),('OkWeather',0),('OkDrummer',1)]) : 0.15,
    frozenset([('OkBureaucracy',0),('OkWeather',1),('OkDrummer',0)]) : 0.34,
    frozenset([('OkBureaucracy',0),('OkWeather',1),('OkDrummer',1)]) : 0.47, #
    frozenset([('OkBureaucracy',1),('OkWeather',0),('OkDrummer',0)]) : 0.54,
    frozenset([('OkBureaucracy',1),('OkWeather',0),('OkDrummer',1)]) : 0.66,
    frozenset([('OkBureaucracy',1),('OkWeather',1),('OkDrummer',0)]) : 0.85,
    frozenset([('OkBureaucracy',1),('OkWeather',1),('OkDrummer',1)]) : 0.99
}
n6.assign_CPT(full_cpt=concert_cpt)
