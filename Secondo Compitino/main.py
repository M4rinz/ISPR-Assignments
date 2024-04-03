# IMPORTS?
from bayes_net import Node, BayesNetwork




#def main() -> None:
print("Welcome!")

#create nodes
n1 = Node(label="primo", node_id = 1)
n2 = Node(label="secondo", node_id = 2)
n3 = Node(label="terzo")

#create bayes net
BN1 = BayesNetwork()

#add nodes to the net
BN1.add_nodes(n1)
BN1.add_nodes(n2)
BN1.add_nodes(n3)

# create arcs
arc1 = (1,2)
arc2 = (1,3)

#add it to the network
BN1.add_arcs(arc1,arc2)

#check that everything is fine
n1.print_attributes()
n2.print_attributes()
n3.print_attributes()
print()

n7 = Node(label="settimo", node_id = 7)
n8 = Node(label="ottavo", node_id = 8)
BN2 = BayesNetwork([n7,n8], [(7,8)])

print(f"Nodes in BN2 are {list(map(lambda x: x.ID, BN2._nodes_list))}")

#if __name__ == "__main__":
#    main()

