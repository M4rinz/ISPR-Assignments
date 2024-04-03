from typing import List, Callable, Tuple
from exceptions import *

class Node():
    def __init__(self, label:str, node_id:int = None):  # Of course, more to come!
        self.label = label
        self.ID = node_id
        self.FS = []    # forward star
        self.BS = []    # backward star

    def set_id(self, node_id:int) -> None:
        self.ID = node_id

    def print_attributes(self) -> None:
        print(f'Node ID: {self.ID}')
        print(f'Node Label: {self.label}')
        print(f'Its children are {self.FS}')
        print(f'Its parents are {self.BS}')

    

class BayesNetwork():
    def __init__(self, 
                 nodes:List[Node],          # maybe not the best idea. We'll see
                 arcs=List[Tuple[int,int]]):   
        self._nodes_list = nodes            # We assume that the nodes are given without the ID. 
                                            # Maybe I'll use the IDs for the topological sorting
        self._arcs_list = arcs
        #TODO: some safety measures?

    def add_node(self, label:str) -> None:
        new_node = Node(label,id)
        self._nodes_list.append(new_node)

    #def add_arc(self, from_node:int, to_node:int) -> None:


    def update_FS_BS(self) -> None:
        for arc in self._arcs_list:
            head, tail = arc
            head_node = [n for n in self._nodes_list if n.ID == head]
            if len(head_node) > 1:
                raise DuplicateNodeID
