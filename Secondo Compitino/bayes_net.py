from typing import List, Callable, Tuple
import exceptions

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
        print()


    def _add_to_star(self, idx:int, S) -> None:
        if idx == self.ID:
            raise exceptions.InvalidArcException(f"Tried to insert a self-loop in node {idx}")
        
        if idx not in S:
            S.append(idx)
        else:
            raise exceptions.InvalidArcException("Redundant arc?")

    def add_child(self, idx:int) -> None:    # Maybe use a list of Nodes directly??
        self._add_to_star(idx, self.FS)

    def add_parent(self, idx:int) -> None:
        self._add_to_star(idx, self.BS)

    

class BayesNetwork():
    def __init__(self, 
                 nodes:List[Node] = [],          # maybe not the best idea. We'll see
                 arcs:List[Tuple[int,int]] = []):   
        self._nodes_list = nodes            # We assume that the nodes are given without the ID. 
                                            # Maybe I'll use the IDs for the topological sorting
        self._arcs_list = arcs
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
                    "the new node will receive automatically the ID", new_id)
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
                tail_node.add_child(head_ID)   

                head_node = head_node.pop()
                head_node.add_parent(tail_ID)

                # update list of arcs
                self._arcs_list.append(arc)
            except exceptions.InvalidArcException as exc:
                print(f'Caught exception with message: {exc}')
                print(f'The arc will be ignored')
                pass
            except exceptions.DuplicateNodeIDError as exc:
                print(exc)
                print("How do you want to deal with it?")
                #TODO deal with it

