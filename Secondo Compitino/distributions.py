import numpy as np
from typing import List, Tuple, Dict

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
                ):
        self.parents = {node.label: node for node in parents_list}
        self.cond_distrib = self.build_cond_distrib(init_dict)  # welcome to the simple affairs' complication office 
        self.n_parents = len(list(self.cond_distrib.keys())[0])  # hopefully there is a clearer way

    def build_cond_distrib(self, 
                           passed_dict:Dict[frozenset[Tuple[str,int]], float]) -> \
                           Dict[frozenset[Tuple], float]:
        '''
        Builds the actual dictionary that is used 
        as CPT data structure.
        In practice, substitute the labels with the actual node
        '''
        init_dict = {}
        for key, value in passed_dict.items():
            new_row = [(self.parents[label],i) for label, i in key]
            new_row = frozenset(new_row)
            init_dict[new_row] = value


        #TODO: check if everything is fine. Else throw exception?
        # What to do?
        return init_dict


    def print_cpt(self) -> None:
        print("To be implemented!")

    # assuming that the RV is Bernoullian
    def sample(self) -> int:
        evidence = [] #np.zeros(self.n_parents)     # I really didn't want to do this
        for parent in self.parents.values():
            #parent = self.parents[i]    # get the i-th parent (is a node)
            parent_sample = parent.distribution.sample() # sample from the parent's distribution. 
                                                    # This should trigger a recursive call
            evidence.append((parent, parent_sample))   # create a tuple with the outcome
            
        # once we have all the evidence, we can sample according to
        # the conditional distribution of the node (as specified by this CPT)

        evidence = frozenset(evidence)  
        p = self.cond_distrib[evidence]
        return int(np.random.random() < p)

    def edit_p(self, 
               assignment:Dict[str,int],
               new_p:float) -> None:

        cpt_row = [(self.parents[label],val) for label, val in assignment.values()]
        cpt_row = frozenset(cpt_row)
        self.cond_distrib[cpt_row] = new_p




    '''
        idea: frozensets as keys!! This might be it!

        {
            {(p1,1),(p2,0),(p3,0)} : val_p,
            ...
            {(p1,1),(p2,1),(p3,1)} : val_p
        }

        aka a Dict[frozenset[Tuple[Node, int]], float]
        type(p1) = Node


        idea2: instead of Tuples[Node, int] (i.e. tuples like (p1,1))
        use Tuple[str, int], that is use the label of the node instead
        of the node itself. At least for the assignment maybe
    '''