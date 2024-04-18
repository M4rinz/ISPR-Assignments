import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from itertools import product
import numpy as np

from bayes_net import Node, BayesNetwork


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
n9 = Node(label="goodPerformance", node_id=9)
n10 = Node(label="concert_held", node_id=10)
n11 = Node(label="concert_success", node_id=11)

# create arcs
arcs_list = [('ok_soprintendenza','ok_bureaucracy')]
arcs_list.append(('sound_check','ok_bureaucracy'))
arcs_list.append(('weather','ok_civilEngineering'))
arcs_list.append(('ok_civilEngineering','ok_bureaucracy'))
arcs_list.append(('ok_drummer','concert_held'))
arcs_list.append(('weather','concert_held'))
arcs_list.append(('ok_drummer','goodPerformance'))
arcs_list.append(('pressCoverage','manyPeople'))
arcs_list.append(('manyPeople','concert_success'))
arcs_list.append(('pressCoverage','concert_success'))
arcs_list.append(('ok_bureaucracy','concert_held'))
arcs_list.append(('concert_held','concert_success'))
arcs_list.append(('concert_held','goodPerformance'))
arcs_list.append(('goodPerformance','concert_success'))


# create bayes net
BN = BayesNetwork([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11], arcs_list)



############################# DISTRIBUTIONS #############################

# create distributions for the unconditional nodes
n1.assign_CPT(p = 0.85)                     # la soprintendenza può non dare l'ok con una probabilità del 15%
n2.assign_CPT(p = [0.8, 0.15, 0.05])        # sound threshold check can be 1 = ok, 2 = too loud, 3 = far too loud
n3.assign_CPT(p = [0.9, 0.08, 0.02])        # weather has a 90% chance of being good, and 2% of being very bad (summer thunderstorms)
n4.assign_CPT(p = 0.85)                     # the drummer can be sick with a 15% chance
n5.assign_CPT(p = 0.75)                     # The press coverage of the event represents how well has the event been advertised.
                                            # Despite the effort of the promoters, the press coverage has a high chance of being unsatisfactory


# create the CPT for the civil engineering
civil_engineer_cpt = {
    frozenset([('weather',1)]) : 0.96,      # With a good weather, the civil engineering office will give the ok with a high chance,
                                            # but unfortunately it's not the only thing that the office takes into account (hence the 96%)
    frozenset([('weather',2)]) : 0.85,      # with the rain, the civil engineering office is less likely to give the ok
    frozenset([('weather',3)]) : 0.70       # A summer thunderstorm gives more problems
}
n6.assign_CPT(full_cpt=civil_engineer_cpt)

# create the CPT for the bureaucracy
bureaucracy_cpt = {}
for sop, eng, sound in product(range(2), range(2), range(1,4)):
    s_tuple = ('ok_soprintendenza',sop)
    e_tuple = ('ok_civilEngineering',eng)
    sound_tuple = ('sound_check',sound)
    assignment = [s_tuple, e_tuple, sound_tuple]
    prob = (
        0.05 +              # base probability        
        sop*0.44 +          # soprintendenza alone contributes by a 44%
        eng*0.19 +          # civil engineering alone contributes by a 19%
        round(10*(3-sound)/105,2) +   # sound alone contributes by at most ~19%, and scales linearly
        0.04*(sop and eng and (sound==1)) +     # in ideal conditions, +4%
        0.03*(sum([sop,(sound in [1,2]),eng]))  # if any two variables are "favorable", +3%
    )
    bureaucracy_cpt[frozenset(assignment)] = prob
'''
Even if everything is fine, there is still a 1% chance that the event is not 
authorized due to a minor bureaucratic issue of some sort 
(not explicitely modeled in this BN). 

By the same token, even if none of the "preconditions" are met, 
there is a 5% chance that the concert is authorized
anyway (because, for example, a high-ranking official or polititian
intervenes in favour of the concert's promoters)
'''
n7.assign_CPT(full_cpt=bureaucracy_cpt)

# create the CPT for the manyPeople r.v.
many_people_cpt = {
    frozenset([('pressCoverage',1)]) : 0.92,    # If the event was very advertised, there is a high chance 
                                                # that many people will have bought a ticket

    frozenset([('pressCoverage',0)]) : 0.74,    # With an unsatisfactory press coverage, 
                                                # the chance that many people buy the tickets is lower,
                                                # but still high enough (becuse the band is famous)
}
n8.assign_CPT(full_cpt=many_people_cpt)


# create the CPT for the goodPerformance r.v.
good_performance_cpt = {}
for drummer,held in product(range(2),range(2)):
    good_performance_cpt[frozenset([('ok_drummer',drummer),('concert_held',held)])] = held*(0.65 + 0.20*drummer)
'''
If the concert isn't held, of course the band can't have a good performance.
The absence of the drummer also impacts on the performance, as the band 
is not used to play without him, and the replacement is not as good.
'''
n9.assign_CPT(full_cpt=good_performance_cpt)


# create the CPT for the concert_held r.v.
concert_held_cpt = {}
for w,b,d in product(range(1,4),range(2),range(2)):
    prob = (
        0.07 +                      # base probability, chosen s.t. 0.07**1.5 > 0.02
        0.6*b +                     # bureaucracy alone impacts for 60% 
        d*(b*0.29 + (1-b)*0.13)     # the impact of the presence of the drummer has been refactored
    )
    prob = prob**(w/4 + 0.75)
    concert_held_cpt[frozenset([('weather',w),('ok_bureaucracy',b),('ok_drummer',d)])] = prob
'''
Raising to the w power keeps the interval [0,1] within itself.
The transformation w -> w/4 + 0.75 maps [1,3] into [1,1.5], which 
gives a moderate decrease to the argument. Higher exponets are too drastic.
The idea is that worse weather corresponds to a higher exponents, and drives 
all the probabilities towards 0.

The idea behind the base probability is that even against all odds,
the organizer have (less than) a 2% probability to organize and perform
the concert anyway. Of course, they'd face sanctions in doing so, due to the
lack of the proper authorizations. 

All in all, the formula inside the prob variable has been hand-crafted
to return the desired results, still witholding some logic.
In particular, the probability of success conditioned to ideal condition is 96%
(not 100% because, of course, there can be an unlucky event ruining everything).
'''
n10.assign_CPT(full_cpt=concert_held_cpt)


# create the CPT for the concert_success r.v.
concert_success_cpt = {}
for held,gp,people,press in product(range(2),range(2),range(2),range(2)):
    list_inside = [('concert_held',held),('manyPeople',people),('goodPerformance',gp),('pressCoverage',press)]
    qty = (
        gp*0.45 +                           # a good performance is key... 
        people*0.1 +                        # ... as well as a vibrant public ...
        people*gp*0.3                       # ... but the real key to success is to have them both!
    )
    qty *= (press+0.1)                      # A good press coverage "amplifies" the factors that 
                                            # contribute to the success of the live performance,
                                            # thus giving the impression of a even better event. 
    concert_success_cpt[frozenset(list_inside)] = held*(qty +0.02)

n11.assign_CPT(full_cpt=concert_success_cpt)


################################### SAMPLING ###################################
N_SAMPLES = 10000


# Sample the Bayesian Network for ok_bureaucracy
ok_bureaucracy = 0
for _ in range(N_SAMPLES):
    ok_bureaucracy += n7.distribution.sample()

print(f'Out of {N_SAMPLES} samples of the Bayesian Network, {ok_bureaucracy} '
        'resulted in the authorization of the concert. ')
print(f'Based on this, the probability of the concert being authorized is {ok_bureaucracy/N_SAMPLES:.3f}')
print()

# Sample the Bayesian Network for concert_held
concert_held = 0
for _ in range(N_SAMPLES):
    concert_held += n10.distribution.sample()

print(f'Out of {N_SAMPLES} samples of the Bayesian Network, {concert_held} '
        'resulted in the concert being held. ')
print(f'Based on this, the probability of the concert being held is {concert_held/N_SAMPLES:.3f}')
print()

# Sample the Bayesian Network for concert_success
concert_success = 0
for _ in range(N_SAMPLES):
    concert_success += n11.distribution.sample()

print(f'Out of {N_SAMPLES} samples of the Bayesian Network, {concert_success} '
        'resulted in a successful concert. ')
print(f'Based on this, the probability of a successful concert is {concert_success/N_SAMPLES:.3f}')
print()
