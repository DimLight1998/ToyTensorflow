import numpy as np


class Node:
    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = []
        self.value = None

        for node in inputs:
            node.outputs.append(self)


class Operation(Node):
    def __init__(self, inputs):
        super().__init__(inputs)

    def compute(self):
        raise NotImplementedError()

    def get_gradient(self, gradient_to_this):
        raise NotImplementedError()


class Placeholder(Node):
    def __init__(self):
        super().__init__([])


class Variable(Node):
    def __init__(self, initial_value):
        super().__init__([])
        self.value = initial_value


class Add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        return x + y

    def get_gradient(self, gradient_to_this):
        shape_x = self.inputs[0].value.shape
        shape_y = self.inputs[1].value.shape
        if len(shape_x) < len(shape_y):
            return [np.sum(gradient_to_this, 0), gradient_to_this]
        if len(shape_x) > len(shape_y):
            return [gradient_to_this, np.sum(gradient_to_this, 0)]

        return [gradient_to_this, gradient_to_this]


class MatMul(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        return x.dot(y)

    def get_gradient(self, gradient_to_this):
        x_value = self.inputs[0].value
        y_value = self.inputs[1].value
        return [gradient_to_this.dot(y_value.T), x_value.T.dot(gradient_to_this)]


class Log(Operation):
    def __init__(self, x):
        super().__init__([x])

    def compute(self, x):
        return np.log(x)

    def get_gradient(self, gradient_to_this):
        x_value = self.inputs[0].value
        return [gradient_to_this / x_value]


class HadamardProduct(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        return x * y

    def get_gradient(self, gradient_to_this):
        x_value = self.inputs[0].value
        y_value = self.inputs[1].value
        return [gradient_to_this * y_value, gradient_to_this * x_value]


class Negative(Operation):
    def __init__(self, x):
        return super().__init__([x])

    def compute(self, x):
        return -x

    def get_gradient(self, gradient_to_this):
        return [-gradient_to_this]


class ReduceSum(Operation):
    def __init__(self, x, axis=None):
        super().__init__([x])
        self.axis = axis

    def compute(self, x):
        return np.sum(x, self.axis)

    def get_gradient(self, gradient_to_this):
        shape = np.array(self.inputs[0].value.shape)
        shape[self.axis] = 1
        new_shape = self.inputs[0].value.shape // shape
        gradient = np.reshape(gradient_to_this, shape)
        return [np.tile(gradient, new_shape)]


class Sigmoid(Operation):
    def __init__(self, x):
        super().__init__([x])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compute(self, x):
        return sigmoid(x)

    def get_gradient(self, gradient_to_this):
        value = self.inputs[0].value
        return [gradient_to_this * sigmoid(value) * (1 - sigmoid(value))]


class Softmax(Operation):
    def __init__(self, x):
        super().__init__([x])

    def compute(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]

    def get_gradient(self, gradient_to_this):
        value = self.value
        return [(gradient_to_this - np.reshape(np.sum(gradient_to_this * value, 1), [-1, 1])) * value]


class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def __init__(self):
                super().__init__([])

            def compute(self):
                gradients_table = compute_gradients(loss)
                for node in gradients_table:
                    if isinstance(node, Variable):
                        node.value -= learning_rate * gradients_table[node]

        return MinimizationOperation()


class Session:
    def run(self, target, feed_dict):
        nodes_sorted = topological_sort(target)
        for node in nodes_sorted:
            if isinstance(node, Placeholder):
                node.value = feed_dict[node]
            elif isinstance(node, Operation):
                node.value = node.compute(*[input_node.value for input_node in node.inputs])
        return target.value


def compute_gradients(target_node):
    gradients_table = {}

    nodes_sorted = topological_sort(target_node)[::-1]
    for node in nodes_sorted:
        if node == target_node:
            gradients_table[node] = 1
        else:
            gradient = 0
            for consumer in node.outputs:
                index = consumer.inputs.index(node)
                gradient += consumer.get_gradient(gradients_table[consumer])[index]
            gradients_table[node] = gradient
    return gradients_table


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


data_0 = np.stack([np.array([+2 * np.random.rand(), +2 * np.random.rand()]) for _ in range(32)])
data_1 = np.stack([np.array([-2 * np.random.rand(), -2 * np.random.rand()]) for _ in range(32)])
data_x = np.vstack((data_0, data_1))
data_y = np.array([[1, 0]] * 32 + [[0, 1]] * 32)

inputs = Placeholder()
labels = Placeholder()
weights = Variable(np.random.rand(2, 2))
bias = Variable(np.random.rand(2))
logits = Add(MatMul(inputs, weights), bias)
predictions = Softmax(logits)
loss = cross_entropy_loss(labels, predictions)
minimization = GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
session = Session()
print(session.run(loss, feed_dict={inputs: data_x, labels: data_y}))
for epoch in range(128):
    session.run(minimization, feed_dict={inputs: data_x, labels: data_y})
    print(session.run(loss, feed_dict={inputs: data_x, labels: data_y}))
print(session.run(weights, feed_dict={inputs: data_x, labels: data_y}))
print(session.run(bias, feed_dict={inputs: data_x, labels: data_y}))
