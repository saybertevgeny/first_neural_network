import numpy
import matplotlib.pyplot


class Paint:

    def __init__(self, filepath):
        self.filepath = filepath
        self.data_list = self.parsefile()

    def parsefile(self):
        data_file = open(self.filepath, 'r')
        data_list = data_file.readlines()
        data_file.close()
        return data_list

    def show(self, row_number):
        all_values = self.data_list[row_number].split(',')
        image_array = numpy.asfarray(all_values[1:]).reshape(28, 28)
        matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
        matplotlib.pyplot.show()
