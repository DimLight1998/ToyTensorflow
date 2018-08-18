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


class Log(Operation):
    def __init__(self, x):
        super().__init__([x])

    def compute(self, x):
        return np.log(x)


class HadamardProduct(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        return x * y


class Negative(Operation):
    def __init__(self, x):
        return super().__init__([x])

    def compute(self, x):
        return -x


class ReduceSum(Operation):
    def __init__(self, x, axis=None):
        self.axis = axis
        return super().__init__([x])

    def compute(self, x):
        return np.sum(x, self.axis)


class Sigmoid(Operation):
    def __init__(self, x):
        super().__init__([x])

    def compute(self, x):
        return 1 / (1 + np.exp(-x))


class Softmax(Operation):
    def __init__(self, x):
        super().__init__([x])

    def compute(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]


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


def cross_entropy_loss(labels, predictions):
    return Negative(ReduceSum(HadamardProduct(labels, Log(predictions))))


a = np.array([[1, 2, 3, 4, 5, 6, 7, 10]])
b = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])

x = Placeholder()
y = Placeholder()
z = Softmax(x)
loss = cross_entropy_loss(y, z)
session = Session()
print(session.run(loss, {x: a, y: b}))
