from NeuralNetwork import NeuralNetwork
import numpy

# количество входных, скрытых и выходных узлов
# 28x28
input_nodes = 784
# 100
hidden_notes = 100
# маркер отвечающий за цифру
output_notes = 10

# коэфициент обучения
learning_rate = 0.3

n = NeuralNetwork(input_nodes, hidden_notes, output_notes, learning_rate)

# загрузить в список тестовый набор данных CSV-файла набора MNIST
training_data_file = open("mnist_train_100.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')
    # масштабировать и сместить входные значения
    inputs = (numpy.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    # создать желаемы целевые значения для 5 это [0,0,0,0,0,0.9,0,0,0,0]
    targets = numpy.zeros(output_notes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

testRecord = training_data_list[38].split(',')
input_test = (numpy.asfarray(testRecord[1:]) / 255 * 0.99) + 0.01
print(n.query(input_test))
