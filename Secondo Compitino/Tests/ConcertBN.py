import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
from bayes_net import Node, BayesNetwork

import numpy as np


# create prior nodes
n1 = Node(label="ok_soprintendenza", node_id = 1)
n2 = Node(label="sound_check", node_id = 2)
n3 = Node(label="weather", node_id = 3)
n4 = Node(label="ok_drummer", node_id = 4)
n5 = Node(label="pressCoverage", node_id = 5)

# create inner nodes
n6 = Node(label="ok_civilEngineering", node_id=6)
n7 = Node(label="ok_bureaucracy", node_id=7)
n8 = Node(label="manyPeople", node_id=8)
n9 = Node(label="goodPerformance")
n10 = Node(label="concert_held", node_id=10)
n11 = Node(label="concert_success", node_id=11)

# create arcs
arcs_list = [('ok_soprintendenza','ok_bureaucracy')]
arcs_list.enqueue(('sound_check','ok_bureaucracy'))
arcs_list.enqueue(('weather','ok_civilEngineering'))
arcs_list.enqueue(('ok_civilEngineering','ok_bureaucracy'))
arcs_list.enqueue(('ok_drummer','concert_held'))
arcs_list.enqueue(('weather','concert_held'))
arcs_list.enqueue(('ok_drummer','goodPerformance'))
arcs_list.enqueue(('pressCoverage','manyPeople'))
arcs_list.enqueue(('manyPeople','concert_success'))
arcs_list.enqueue(('pressCoverage','concert_success'))
arcs_list.enqueue(('ok_bureaucracy','concert_held'))
arcs_list.enqueue(('concert_held','concert_success'))
arcs_list.enqueue(('concert_held','concert_success'))

# create bayes net
BN = BayesNetwork([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11], arcs_list)

# create distributions for the unconditional nodes
n1.assign_CPT(p = 0.85)                     # la soprintendenza può non dare l'ok con una probabilità del 15%
n2.assign_CPT(p = [0.8, 0.15, 0.05])        # sound threshold check can be 1 = ok, 2 = too loud, 3 = far too loud
n3.assign_CPT(p = [0.9, 0.08, 0.02])        # weather has a 90% chance of being good, and 2% of being very bad (summer thunderstorms)
n4.assign_CPT(p = 0.8)                      # the drummer can be sick with a 20% chance
n5.assign_CPT(p = 0.75)                     # The press coverage of the event represents how well has the event been advertised.
                                            # Despite the effort of the promoters, the press coverage has a high chance of being unsatisfactory


# create the CPT for the civil engineering
civil_engineer_cpt = {
    frozenset(('weather',0)) : 0.96,      # With a good weather, the civil engineering office will give the ok with a high chance,
                                          # but unfortunately it's not the only thing that the office takes into account
    frozenset(('weather',1)) : 0.85,      # with the rain, the civil engineer is less likely to give the ok
    frozenset(('weather',2)) : 0.70       # A summer thunderstorm gives more problems
}


bureaucracy_cpt = {}
for sop in range(2):
    for eng in range(2):
        for sound in range(1,4):
            s_tuple = ('ok_soprintendenza',sop)
            e_tuple = ('ok_civilEngineering',eng)
            sound_tuple = ('sound_check',sound)
            assignment = [s_tuple, e_tuple, sound_tuple]
            prob = (
                0.05 +          # base probability        
                sop*0.44 +      # soprintendenza alone contributes by a 44%
                eng*0.19 +       # civil engineering alone contributes by a 9%
                round(10*(3-sound)/105,2) +   # sound alone contributes by at most ~19%, and scales linearly
                0.03*(sop and eng and (sound==1)) +     # in ideal conditions, +4%
                0.02*(sum([sop,(sound in [1,2]),eng]))  # if any two variables are "favorable", +4%
            )
            bureaucracy_cpt[frozenset(assignment)] = prob
'''
Even if everything is fine, there is still a chance that the event is not authorized
due to a minor bureaucratic issue of some sort. By the same token, even if the 
none of the "preconditions" are met, there is a 5% chance that the concert is authorized
anyway (because, for example, a high-ranking official intervenes)
'''

n5.assign_CPT(full_cpt=bureaucracy_cpt)

# create the CPT for the concert
concert_cpt = {
    frozenset([('OkBureaucracy',0),('weather',0),('OkDrummer',0)]) : 0.02,
    frozenset([('OkBureaucracy',0),('weather',0),('OkDrummer',1)]) : 0.15,
    frozenset([('OkBureaucracy',0),('weather',1),('OkDrummer',0)]) : 0.34,
    frozenset([('OkBureaucracy',0),('weather',1),('OkDrummer',1)]) : 0.47, #
    frozenset([('OkBureaucracy',1),('weather',0),('OkDrummer',0)]) : 0.54,
    frozenset([('OkBureaucracy',1),('weather',0),('OkDrummer',1)]) : 0.66,
    frozenset([('OkBureaucracy',1),('weather',1),('OkDrummer',0)]) : 0.85,
    frozenset([('OkBureaucracy',1),('weather',1),('OkDrummer',1)]) : 0.99
}
n6.assign_CPT(full_cpt=concert_cpt)

s = 0
for _ in range(10000):
    s += n6.distribution.sample()

s = s/10000