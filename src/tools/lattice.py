from typing import Dict
from matplotlib import pyplot as plt
from graphviz import Digraph 
class Lattice:
    def __init__(self):
        self.G = Digraph(comment='Lattice', strict=True) 
        self.layers = {}
    def addNode(self, node, parent1, parent2, level):
        self.G.add_node()
        self.G.node(serial(t), label(t))
        self.G.edge(serial(parent1), serial(t))
        self.G.edge(serial(parent2), serial(t))
        if level in self.layers:
            self.layers[level] = []
        self.layers[level].append(serial(t))
    def genPlot(self):
        for l in self.layers:
            with self.G.subgraph(name=f'layer%d' % {l}, node_attr={'shape': 'box'}) as c:
                c.attr(rank='same')
                for _node in self.layers[l]:
                    c.node(serial(_node))
        self.G



def serial(self, obj: Dict) -> str:
    return str(hash(str(obj)))
def label(obj: Dict) -> str:    
    return str(obj) 



# # %%
# from graphviz import Digraph 
# import copy
# from itertools import combinations
# from typing import Dict



# layer2 = [{'A':'a1', 'B':'b1'}, {'B':'b2', 'C':'c1'}]



# dot = Digraph(strict=True)

# for t in layer1:
#     dot.node(serial(t), label(t))

# layer2 = []
# for comb in combinations(layer1, 2):
#     t = copy.deepcopy(comb[1])
#     t.update(comb[0])
#     if len(t) < 2:
#         continue
#     layer2.append(t)
#     dot.node(serial(t), label(t))
#     dot.edge(serial(comb[0]), serial(t))
#     dot.edge(serial(comb[1]), serial(t))

# layer3 = []
# for comb in combinations(layer2, 2):
#     t = copy.deepcopy(comb[1])
#     t.update(comb[0])
#     if len(t) < 3:
#         continue
#     layer3.append(t)
#     dot.node(serial(t), label(t))
#     dot.edge(serial(comb[0]), serial(t))
#     dot.edge(serial(comb[1]), serial(t))

# with dot.subgraph(name='child0', node_attr={'shape': 'box'}) as c:
#     c.attr(rank='same')
#     for n in layer1:
#         c.node(serial(t))
# with dot.subgraph(name='child1', node_attr={'shape': 'box'}) as c:
#     c.attr(rank='same')
#     for n in layer2:
#         c.node(serial(t), label(t))
# with dot.subgraph(name='child2', node_attr={'shape': 'box'}) as c:
#     c.attr(rank='same')
#     for n in layer3:
#         c.node(serial(t), label(t))

 
# print(dot)
# dot

# # %%
