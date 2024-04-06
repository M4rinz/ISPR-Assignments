import numpy as np
from typing import List, Tuple, Dict
import exceptions

class PriorBernoulli():
    def __init__(self, p:float):
        self._p = p

    def set_p(self, new_p:float) -> None:
        self._p = new_p

    def get_p(self) -> float:
        return self._p
    
    def sample(self) -> int:
        return int(np.random.random() < self._p)
    

class CPT():
    def __init__(self, 
                init_dict:Dict[frozenset[Tuple[str,int]], float],   # WIP: dictionary with the initializations
                parents_list:List,  # the idea is to just pass the BS
                node_label:str
                ):
        self.owner_node_label = node_label
        self.parents = {node.label: node for node in parents_list}
        try:
            self.cond_distrib = self.build_cond_distrib(init_dict)  # welcome to the simple affairs' complication office 
        except exceptions.WrongAssignment as exc:
            print(f'The CPT assignment is {exc.args[0]['status']}complete at row {exc.args[0]['row']}')
            #print(exc)
            print('The CPT will be automatically set to None.') 
            print('Please destroy this object and try again')
            self.cond_distrib = None
            # can really nothing be done for this issue?

    def build_cond_distrib(self, 
                           passed_dict:Dict[frozenset[Tuple[str,int]], float]) -> \
                           Dict[frozenset[Tuple], float]:
        '''
        Builds the actual dictionary that is used 
        as CPT data structure.
        In practice, substitute the labels with the actual node
        '''
        init_dict = {}
        for i, (key, value) in enumerate(passed_dict.items(), 1):
            new_row = [(self.parents[label],i) for label, i in key]

            checkSet = set()
            for tuple in key:
                checkSet.add(tuple[0])
            if set(self.parents.keys()) != checkSet:
                if set(self.parents.keys()) < checkSet:
                    status = 'over'
                else:
                    status = 'under'
                raise exceptions.WrongAssignment({'row':i, 'status': status})

            new_row = frozenset(new_row)
            init_dict[new_row] = value

        return init_dict


    def print_cpt(self) -> None:
        for key, value in self.cond_distrib.items():
            assignments = [f"{node.label}={val}" for node, val in key]
            print(f"P({self.owner_node_label}=1 | {', '.join(assignments)}) = {value}")

    def set_p(self, 
            assignment:Dict[str,int],
            new_p:float) -> None:
        cpt_row = [(self.parents[label],val) for label, val in assignment.values()]
        cpt_row = frozenset(cpt_row)
        self.cond_distrib[cpt_row] = new_p

    def get_p(self, 
              assignment:Dict[str,int]) -> float:
        cpt_row = [(self.parents[label],i) for label, i in assignment.values()]
        cpt_row = frozenset(cpt_row)
        return self.cond_distrib[cpt_row]

    # assuming that the RV is Bernoullian
    def sample(self) -> int:
        evidence = []
        for parent in self.parents.values():
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
        return int(np.random.random() < p)
