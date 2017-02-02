
from TheANNXor.Neuron import Neuron

#This class is not necessary in this particular program because its static but
#good to have it because it is always used in other cases

class Layer:
    def __init__(self):
        super().__init__()
        self.neurons = []

    def add_neuron_to_layer(self, Neuron):
        self.neurons.append(Neuron)
        # print(self.neurons)

    def get_neuron_from_layer(self, index):
        return self.neurons[index]

    def get_neurons(self):
        return self.neurons

    def myTestMet(self):
        print("layeer test met")

    def __repr__(self):
        return "Layer({})".format(self.neurons)



