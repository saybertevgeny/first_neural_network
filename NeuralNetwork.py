import numpy
import scipy.special


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # количество узлов в слоях
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate
        # матрица весовых коэфициентов от входного к скрытому слою
        # self.wih = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.wih = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        # матрица весовых коэфициентов от скрытого к выходному слою
        # self.woh = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        self.woh = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        self.activation_function = lambda x: scipy.special.expit(x)

    # тренировка сети
    def train(self, input_list, target_list):
        # преобразовать список входных значений в вухмерный массив
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # рассчитать входящие сигналы ждя скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигнали для выходного слоя
        final_inputs = numpy.dot(self.woh, hidden_outputs)
        # рассчитать входящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # рассчитаем ошибку = целевое значение - фактическое значение
        output_errors = targets - final_outputs

        # ошибки скрытого слоя
        # распределенные пропорционально весовым коэфициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.woh.T, output_errors)

        self.woh += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                   numpy.transpose(hidden_outputs))
        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                                   numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # выходные сигналы
        final_inputs = numpy.dot(self.woh, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
