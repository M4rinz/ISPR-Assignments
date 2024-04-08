import numpy as np
from typing import List, Tuple, Dict
import exceptions

from BNTypes import P, PassedConditions

class PriorBernoulli():
    def __init__(self, p:P):
        self._p = p

    def set_p(self, new_p:P) -> None:
        self._p = new_p

    def get_p(self) -> P:
        return self._p
    
    def sample(self) -> int:
        return int(np.random.random() < self._p)
    

class PriorCategorical():
    def __init__(self, ps:P):
        if sum(ps) != 1 or any(p<0 for p in ps):
            print("The vector of event probabilities should be a stochastic vector.")
            print("A vector of zeros will be assigned. Please call the set_pvec method")
            self._pvec = [0 for _ in ps]
        else:
            self._pvec = ps


    def set_pvec(self, new_pvec:P) -> None:
        self._pvec = new_pvec

    def get_pvec(self) -> P:
        return self._pvec
    
    def sample(self) -> int:
        """Samples from the categorical distribution
            according to the classes' probabilities
            (assumed to be integers in {1,...,n})

        Returns:
            class: the sampled class
        """
        r = np.random.random()
        prev = 0
        for i,p_i in enumerate(self._pvec,1):
            if r < p_i+prev:
                return i
            prev += p_i

    

class CPT():
    def __init__(self, 
                init_dict:Dict[PassedConditions, P],   # WIP: dictionary with the initializations
                parents_list:List,  # the idea is to just pass the BS
                node_label:str
                ):
        self._owner_node_label = node_label
        self._parents = {node.label: node for node in parents_list}
        try:
            self.cond_distrib = self.build_cond_distrib(init_dict)  # welcome to the simple affairs' complication office 
        except exceptions.WrongAssignment as exc:
            print(f'The CPT assignment is {exc.args[0]['status']}complete at row {exc.args[0]['row']}')
            print('The CPT will be automatically set to None.') 
            print('Please call the build_cond_distrib method')
            self.cond_distrib = None
            # can really nothing be done for this issue?

    def build_cond_distrib(self, 
                           passed_dict:Dict[PassedConditions, P]) -> \
                           Dict[frozenset[Tuple], P]:
        '''
        Builds the actual dictionary that is used 
        as CPT data structure.
        In practice, substitute the labels with the actual node
        '''
        init_dict = {}
        for i, (key, value) in enumerate(passed_dict.items(), 1):
            new_row = [(self._parents[label],i) for label, i in key]

            checkSet = set()
            for tuple in key:
                checkSet.add(tuple[0])
            if set(self._parents.keys()) != checkSet:
                if set(self._parents.keys()) < checkSet:
                    status = 'over'
                else:
                    status = 'under'
                raise exceptions.WrongAssignment({'row':i, 'status': status})

            new_row = frozenset(new_row)
            init_dict[new_row] = value

        return init_dict

    # WIP: Again, we're assuming bernoullian rv
    def print_cpt(self) -> None:
        for key, value in self.cond_distrib.items():
            assignments = [f"{node.label}={val}" for node, val in key]
            print(f"P({self._owner_node_label}=1 | {', '.join(assignments)}) = {value}")

    def set_p(self, 
            assignment:Dict[str,int],
            new_p:float) -> None:
        cpt_row = [(self._parents[label],val) for label, val in assignment.values()]
        cpt_row = frozenset(cpt_row)
        self.cond_distrib[cpt_row] = new_p

    def get_p(self, 
              assignment:Dict[str,int]) -> P:
        cpt_row = [(self._parents[label],i) for label, i in assignment.values()]
        cpt_row = frozenset(cpt_row)
        return self.cond_distrib[cpt_row]

    def sample(self) -> int:
        def innerSample(p:P) -> int:
            start = 1
            if isinstance(p,float):
                p = [1-p, p]
                start = 0
            r = np.random.random()
            prev = 0 
            for i, p_i in enumerate(p,start):
                if r < p_i + prev:
                    return i
                prev += p_i

        evidence = []
        for parent in self._parents.values():
            parent_sample = parent.distribution.sample()    # sample from the parent's distribution. 
                                                            # This should trigger a recursive call
            # create a tuple with the outcome and add it to the 
            # evidence we have
            evidence.append((parent, parent_sample))
            
        # once we have all the evidence, we can sample according to
        # the conditional distribution of the node (as specified by this CPT)

        # Since we use frozensets as keys, sampling order doesn't matter

        evidence = frozenset(evidence)  
        p = self.cond_distrib[evidence]

        return innerSample(p)
