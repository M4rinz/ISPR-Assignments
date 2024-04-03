from bayes_net import Node, BayesNetwork
import io
import sys

#create nodes
n1 = Node(label="primo", node_id = 1)
n2 = Node(label="secondo", node_id = 2)
n3 = Node(label="terzo")

#create bayes net
BN1 = BayesNetwork()

#add nodes to the net
BN1.add_nodes(n1)
BN1.add_nodes(n2)

output = io.StringIO()
old_stdout = sys.stdout

sys.stdout = output

BN1.add_nodes(n3)

sys.stdout = old_stdout

# fix this!! Continue!
print("output = ", output.getvalue())

# create arcs
arc1 = (1,2)
arc2 = (1,3)

#add it to the network
BN1.add_arcs(arc1,arc2)

#check that everything is fine
assert set(map(lambda node: node.ID, n1.FS)) == {2,3}
assert set(map(lambda node: node.ID, n1.BS)) == set()


assert set(map(lambda node: node.ID, n2.BS)) == {1}
assert set(map(lambda node: node.ID, n2.FS)) == set()

assert set(map(lambda node: node.ID, n3.BS)) == {1}
assert set(map(lambda node: node.ID, n3.FS)) == set()


n7 = Node(label="settimo", node_id = 7)
n8 = Node(label="ottavo", node_id = 8)
BN2 = BayesNetwork([n7,n8], [(7,8)])

assert set(map(lambda x: x.ID, BN2._nodes_list)) == {7,8}



print("Test completed successfully")

