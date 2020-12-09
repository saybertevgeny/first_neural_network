from NeuralNetwork import NeuralNetwork

# количество входных, скрытых и выходных узлов
input_nodes = 3
hidden_notes = 3
output_notes = 3

# коэфициент обучения
learning_rate = 0.3

n = NeuralNetwork(input_nodes, hidden_notes, output_notes, learning_rate)
output_notes = n.query([1.0, 0.5, -1.5])
print("output_notes = ", output_notes)


