import numpy as np


class Node:
    def __init__(self, inputs):
        self.inputs = inputs


class Operation(Node):
    def __init__(self, inputs):
        super().__init__(inputs)

    def compute(self):
        raise NotImplementedError()


class Placeholder(Node):
    def __init__(self):
        super().__init__([])


class Variable(Node):
    def __init__(self, initial_value):
        self.value = initial_value
        super().__init__([])


class Add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        return x + y


class MatMul(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        return x.dot(y)


class Session:
    def run(self, target, feed_dict):
        nodes_sorted = topological_sort(target)
        results = {node: None for node in nodes_sorted}
        for node in nodes_sorted:
            if isinstance(node, Variable):
                results[node] = node.value
            elif isinstance(node, Placeholder):
                results[node] = feed_dict[node]
            elif isinstance(node, Operation):
                results[node] = node.compute(*[results[input_node] for input_node in node.inputs])
        return results[target]


def topological_sort(node):
    sorted_nodes = []
    queue = [node]
    while len(queue) > 0:
        head = queue.pop(0)
        for pre_node in head.inputs:
            if pre_node not in sorted_nodes and pre_node not in queue:
                queue.append(pre_node)
        sorted_nodes.append(head)
    return sorted_nodes[::-1]


a = np.ones((2, 3))
b = np.ones((3, 1))

x = Placeholder()
y = Placeholder()
z = MatMul(x, y)
session = Session()
print(session.run(z, {x: a, y: b}))
