import io
from platform import node
import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
from bayes_net import Node, BayesNetwork

#create initial nodes
n1 = Node(label="primo", node_id = 1)
n2 = Node(label="secondo", node_id = 2)
n3 = Node(label="terzo", node_id=3)

# create initial arcs
arc1 = (1,2)
arc2 = (1,3)

# create the BN with all this info
BN = BayesNetwork([n1,n2,n3], [arc1,arc2])

# add a further node. Graph is still acyclic
BN.add_arcs((2,3))

n4 = Node(label="quarto", node_id = 4)
arc3 = (3,4)

# add the new node.
BN.add_nodes(n4)
BN.add_arcs(arc3)

# Create a new StringIO object
output = io.StringIO()
# Save the current value of sys.stdout so we can restore it later
old_stdout = sys.stdout
# Redirect sys.stdout to our StringIO object
sys.stdout = output

# operation that issues the print: add cycle-inducing arc
BN.add_arcs((4,1))

# Restore standard output to the original one
sys.stdout = old_stdout

assert output.getvalue() == '''The arc from node quarto to node primo creates an oriented cycle in the graph. This is not allowed.
This arc will be ignored.
'''
assert set(map(lambda nodo: nodo.ID, n4.FS)) == set()
assert set(map(lambda nodo: nodo.ID, n1.BS)) == set()

print("Test completed successfully")