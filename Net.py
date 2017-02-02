
from TheANNXor.Neuron import Neuron
from TheANNXor.Layer import Layer


class Synapse(object):
    def __init__(self, weight, neuron_left, neuron_right):
        self.weight = weight
        self.deltaweight = None
        self.neuron_left = neuron_left
        self.neuron_right = neuron_right

    def __repr__(self):
        return "Synapse(weight:{}, \n" \
               "    left:'{}',\n" \
               "    right:'{}'\n)".format(self.weight, self.neuron_left, self.neuron_right)




class Net:
    def __init__(self):
        super().__init__()

        #initital variables
        self.inititial_weight_vals = 0.125


        #MUTABLE : CHANGE CONTENT OF OBJECT WITHOUT CHANGING IDENTITY
        self.layers = []
        self.synapsesl0l1 = {}
        self.synapsesl1l2 = {}


        self.build_all_layers_and_neurons()
        self.build_all_synapses()

        self.expected = 0.000
        self.error = 0.000

    def build_all_layers_and_neurons(self):
        self.layers.append(Layer())
        self.layers[0].add_neuron_to_layer(Neuron(0, (100, 225), 0.000, 0.000))
        self.layers[0].add_neuron_to_layer(Neuron(1, (100, 325), 0.000, 0.000))
        # we'll add the bias neuron because there should be one here but no calculations on it for now
        self.layers[0].add_neuron_to_layer(Neuron(2, (100, 425), 0.000, 0.000))

        self.layers.append(Layer())
        self.layers[1].add_neuron_to_layer(Neuron(0, (400, 125), 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(1, (400, 225), 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(2, (400, 325), 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(3, (400, 425), 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(4, (400, 525), 0.000, 0.000))

        self.layers.append(Layer())
        self.layers[2].add_neuron_to_layer(Neuron(0, (700, 275), 0.000, 0.000))
        self.layers[2].add_neuron_to_layer(Neuron(1, (700, 375), 0.000, 0.000))

    def build_all_synapses(self):
        #synapses between l0 and l1
        # from-to nomenclature
        for i in range(0, 2):
            nl0 = self.get_neuron(0, i)
            for j in range(0, 4):
                nl1 = self.get_neuron(1, j)
                self.synapsesl0l1[i, j] = Synapse(self.inititial_weight_vals, nl0, nl1)

        # synapses between l1 and l2
        for i in range(0, 4):
            nl1 = self.get_neuron(1, i)
            for j in range(0, 1):
                nl2 = self.get_neuron(2, j)
                self.synapsesl1l2[i, j] = Synapse(self.inititial_weight_vals, nl1, nl2)

    def get_weight(self, layerIndex, neuronIndex1, neuronIndex2):
        if layerIndex == 0:
            return getattr(self.synapsesl0l1[0, 0], 'weight')
        else:
            return getattr(self.synapsesl1l2[1, 0], 'weight')

    def get_neuron(self, layerIndex, neuronIndex):
        return self.layers[layerIndex].get_neuron_from_layer(neuronIndex)
        # self.layers[index]

        # print("Net, getting: " + index)
        # self.get_neuron_from_layer(index)

    def get_coordinates(self, Neuron):
        return getattr(Neuron, 'coordinatesMidpoint')
        # return Neuron.coordinatesMidpoint

    def set_input(self, Neuron, inputn):
        setattr(Neuron, 'inputn', inputn)
        # Neuron.set_input(inputn)

    def get_input(self, Neuron):
        return float(getattr(Neuron, 'inputn'))
        # return float(Neuron.inputn)

    def set_output(self, Neuron, output):
        setattr(Neuron, 'output', output)
        # Neuron.set_outputs(output)

    def get_output(self, Neuron):
        return float(getattr(Neuron, 'output'))
        # return Neuron.get_outputs()

    # def set_output_weights(self, Neuron, o_weights):
    #     Neuron.set_weights(o_weights)
    #
    # def get_output_weights(self, Neuron):
    #     return Neuron.get_output_weights()

    def convertToString(self):
        return str(self.layers[0])

    def myNetMethod(self):
        w2 = self.synapsesl0l1[0]

    def get_layer(self, index):
        return self.layers[index]

    def setJo(self, Neuron, inputs):
        Neuron.set_inputs(inputs)





