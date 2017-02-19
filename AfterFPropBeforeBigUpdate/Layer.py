
class Layer:
    def __init__(self):
        super().__init__()
        self.neurons = []

    def add_neuron_to_layer(self, Neuron):
        self.neurons.append(Neuron)

    def get_neuron_from_layer(self, index):
        return self.neurons[index]

    def __repr__(self):
        return "Layer({})".format(self.neurons)

