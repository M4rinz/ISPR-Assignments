from typing import List, Callable, Tuple
import exceptions
from distributions import Bernoulli

class Node():
    def __init__(self, label:str, 
                 node_id:int = None,
                 distrib = None ):  # Of course, more to come!
        self.label = label
        self.ID = node_id
        self.FS = []    # forward star
        self.BS = []    # backward star
        self.distribution = distrib

    def set_id(self, node_id:int) -> None:
        self.ID = node_id

    def print_attributes(self) -> None:
        print()
        print('--------------------------------------------------')
        print(f'{f"Node ID: {self.ID}":^25} | {f"Node Label: {self.label}":^25}')
        print('Its parents are:')
        for i in range(0, len(self.BS), 2):
            print(f'\tID: {self.BS[i].ID} L: {self.BS[i].label:5}', end='')
            if i + 1 < len(self.BS):
                print(f',  ID: {self.BS[i+1].ID} L: {self.BS[i+1].label}')
            else:
                print()
        print('Its children are:')
        for i in range(0, len(self.FS), 2):
            print(f'\tID: {self.FS[i].ID} L: {self.FS[i].label}', end='')
            if i + 1 < len(self.FS):
                print(f',  ID: {self.FS[i+1].ID} L: {self.FS[i+1].label}')
            else:
                print()
        print('--------------------------------------------------')


    def _add_to_star(self, toAdd, S) -> None:
        if toAdd.ID == self.ID:
            raise exceptions.InvalidArcException(f"Tried to insert a self-loop in node whose ID is {toAdd.ID}")
        
        if toAdd not in S:
            S.append(toAdd)
        else:
            raise exceptions.InvalidArcException("Redundant arc?")

    def add_child(self, node) -> None:    # Maybe use a list of Nodes directly??
        self._add_to_star(node, self.FS)

    def add_parent(self, node) -> None:
        self._add_to_star(node, self.BS)

    


    

class BayesNetwork():
    def __init__(self, 
                 nodes:List[Node] = [],          # maybe not the best idea. We'll see
                 arcs:List[Tuple[int,int]] = []):   
        self._nodes_list = []          # We assume that the nodes are given with the ID. 
                                        # Maybe I'll use the IDs for the topological sorting
        self._arcs_list = []

        if nodes != []: 
            self.add_nodes(*tuple(nodes))
        if arcs != []: 
            self.add_arcs(*tuple(arcs))
        #TODO: some more safety measures?

    def get_nodes_number(self) -> int:
        return len(self._nodes_list)
    

    def add_nodes(self, *args) -> None:
        for node in args:
            if node.ID is None:
                new_id = self.get_nodes_number()+1
                print("Trying to insert a node without an ID")
                print("For the moment, this is not OK...")
                print("Based on the current number of nodes in the network,"
                    " the new node will receive automatically the ID", new_id)
                node.set_id(new_id)
            self._nodes_list.append(node)

    #def add_arc(self, from_node:int, to_node:int) -> None:

    def add_arcs(self, *args) -> None:
        for arc in args: 
            try:
                #TODO support for arcs given the node's ID?
                tail_ID, head_ID = arc
                tail_node = [n for n in self._nodes_list if n.ID == tail_ID]
                head_node = [n for n in self._nodes_list if n.ID == head_ID]

                len_tail = len(tail_node)
                len_head = len(head_node)

                if len_tail > 1:
                    raise exceptions.DuplicateNodeIDError(f"Two nodes in the network have the same ID {tail_node}")
                if len_head > 1:
                    raise exceptions.DuplicateNodeIDError(f"Two nodes in the network have the same ID {head_node}")
                
                # This thing here shouldn't be needed. Therefore is commented
                #if len_tail == 0 or len_head == 0:      
                #    if len_tail == 0 and len_head == 0:
                #        print("Trying to add an arc between TWO non existing nodes")
                #        print("We don't like disconnected graphs. This arc will be ignored")
                #        continue
                #    elif len_tail == 0:
                #        print(f"The node with ID {tail_ID} (supposedly the tail node for the arc) is not in the network.\
                #               It will be created and added.")
                #        new_node = Node(label="AUTO_ADDED", node_id = tail_ID )
                #    else:
                #        print(f"The node with ID {head_ID} (supposedly the head node for the arc) is not in the network.\
                #               It will be created and added.")
                #        new_node = Node(label="AUTO_ADDED", node_id = head_ID )
                #    self.add_nodes(new_node)

                # update forward and backward stars of nodes
                tail_node = tail_node.pop()
                head_node = head_node.pop()

                tail_node.add_child(head_node)   
                head_node.add_parent(tail_node)

                # update list of arcs
                self._arcs_list.append(arc)
            except exceptions.InvalidArcException as exc:
                print(f'Caught exception with message: {exc}')
                print('The arc will be ignored')
                pass
            except exceptions.DuplicateNodeIDError as exc:
                print(exc)
                print("How do you want to deal with it?")
                #TODO deal with it

    
    #def check_oriented_loops(self) -> bool:
        # Let's avoid it, maybe
