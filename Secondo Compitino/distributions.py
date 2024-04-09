import numpy as np
from typing import List, Tuple, Dict
import exceptions

from BNTypes import P, PassedConditions

class Prior():
    def __init__(self, p:P):
        if isinstance(p,float):
            self._distrib_name = 'Bernoulli'
        elif isinstance(p,List):
            self._distrib_name = 'Categorical'
        else:
            print("Warning: invalid datatype for probability masses")
            
        self.set_pvec(p)

    def set_pvec(self, new_p:P) -> None:
        if isinstance(new_p,float):
            if new_p < 0 or new_p > 1:
                print(f"p = {new_p} is not a valid probability.")
                print("p will be set to 0.5. Please call the set_pvec method with a valid value")
                self._pvec = [0.5,0.5]
            else:
                self._pvec = [1-new_p,new_p]
        else:
            if sum(new_p) != 1 or any(p_j<0 for p_j in new_p):
                print("The vector of event probabilities should be a stochastic vector.")
                print("A vector of zeros will be assigned. Please call the set_pvec method with valid values")
                self._pvec = [0 for _ in new_p]
            else:
                self._pvec = new_p

    def get_pvec(self) -> P:
        if self._distrib_name == 'Bernoulli':
            return self._pvec[1]
        else:
            return self._pvec
        
    def get_distribution_name(self) -> str:
        return self._distrib_name
    
    def print_distribution(self) -> None:
        if self._distrib_name == 'Bernoulli':
            print(f"P(X=1) = {self._pvec[1]}")
        else:
            for i, p in enumerate(self._pvec, start=1):
                print(f"P(X={i}) = {p}")
    
    def sample(self) -> int:
        """Categorical prior: samples from the categorical distribution
            according to the classes' probabilities
            (assumed to be integers in {1,...,n})

            Bernoullian prior: samples from the bernoullian distribution
            with parameter p (output can be 0 or 1)

        Returns:
            int: the sampled class
        """
        if self._distrib_name == 'Bernoulli':
            start = 0
        else:
            start = 1
        r = np.random.random()
        prev = 0
        for i,p_i in enumerate(self._pvec,start):
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

            self._distribution_name = 'Conditional'
            ty_p = list(init_dict.values())[0]
            if isinstance(ty_p,List):
                self._distribution_name += ' Categorical'
            elif isinstance(ty_p,float):
                self._distribution_name += ' Bernoulli'
            else:
                self._distribution_name = ''
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
    
    def get_distribution_name(self) -> str:
        return self._distribution_name

    def print_cpt(self) -> None:
        for key, value in self.cond_distrib.items():
            assignments = [f"{node.label} = {val}" for node, val in key]
            plusOne = ''
            if 'Bernoulli' in self._distribution_name:
                plusOne = '=1' 
            print(f"P({self._owner_node_label}{plusOne} | {', '.join(assignments)}) = {value}")

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
