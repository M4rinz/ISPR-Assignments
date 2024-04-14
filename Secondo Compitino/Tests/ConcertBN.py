import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
from bayes_net import Node, BayesNetwork

import numpy as np


# create prior nodes
n1 = Node(label="ok_soprintendenza", node_id = 1)
n2 = Node(label="sound_check", node_id = 2)
n3 = Node(label="ok_weather", node_id = 3)
n4 = Node(label="ok_drummer", node_id = 4)
n5 = Node(label="pressCoverage", node_id = 5)

# create inner nodes
n6 = Node(label="ok_civilEngineer", node_id=6)
n7 = Node(label="ok_bureaucracy", node_id=7)
n8 = Node(label="manyPeople", node_id=8)
n9 = Node(label="goodPerformance")
n10 = Node(label="concert_held", node_id=10)
n11 = Node(label="concert_success", node_id=11)

# create arcs
a1 = ('ok_soprintendenza','ok_bureaucracy')
a2 = ('sound_check','ok_bureaucracy')
a3 = ('ok_weather','ok_civilEngineer')
a4 = ('ok_civilEngineer','ok_bureaucracy')
a5 = ('ok_drummer','concert_held')
a6 = ('ok_weather','concert_held')
a7 = ('ok_drummer','goodPerformance')
a8 = ('pressCoverage','manyPeople')
a9 = ('manyPeople','concert_success')
a10 = ('pressCoverage','concert_success')
a11 = ('ok_bureaucracy','concert_held')
a12 = ('concert_held','concert_success')
a13 = ('concert_held','concert_success')

# create bayes net
BN = BayesNetwork([n1,n2,n3,n4,n5,n6], [a1,a2,a3,a4,a5])

# create distributions
n1.assign_CPT(p = 0.85)     # la soprintendenza può non dare l'ok con una probabilità del 15%
n2.assign_CPT(p = [0.8, 0.15, 0.05])       # sound threshold check can be 1 = ok, 2 = too loud, 3 = far too loud
n3.assign_CPT(p = 0.9)      # weather has a 10% chance of being bad
n4.assign_CPT(p = 0.8)      # the drummer can be sick with a 20% chance

# create the CPT for the bureaucracy
bureaucracy_cpt = {
    frozenset([('OkSoprintendenza',0),('sound_check',3)]) : 0.05,
    frozenset([('OkSoprintendenza',0),('sound_check',2)]) : 0.10,
    frozenset([('OkSoprintendenza',0),('sound_check',1)]) : 0.25,
    frozenset([('OkSoprintendenza',1),('sound_check',3)]) : 0.40,
    frozenset([('OkSoprintendenza',1),('sound_check',2)]) : 0.75,
    frozenset([('OkSoprintendenza',1),('sound_check',1)]) : 0.95
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

s = 0
for _ in range(10000):
    s += n6.distribution.sample()

s = s/10000