import numpy

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        #матрица весовых коэфициентов от входного к скрытому слою
        self.wih = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        #матрица весовых коэфициентов от скрытого к выходному слою
        self.woh = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

    def train(self):
        pass

    def query(self):
        pass
