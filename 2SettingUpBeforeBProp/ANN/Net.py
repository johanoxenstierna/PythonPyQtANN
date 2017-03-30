
import math
from ANN.TrainingData import TrainingData

from ANN.Layer import Layer
from ANN.Neuron import Neuron
from ANN.TrainingData import TrainingData

class Synapse:
    def __init__(self, weight, neuron_left, neuron_right):
        self.weight = weight
        self.deltaweight = 0.000
        self.neuron_left = neuron_left
        self.neuron_right = neuron_right

    def __repr__(self):
        return "Synapse(weight:{}, \n" \
               "    left:'{}',\n" \
               "    right:'{}'\n)".format(self.weight, self.neuron_left, self.neuron_right)



class Net:
    def __init__(self):
        super().__init__()

        self.m_training_data = TrainingData()

        self.initial_weight_vals = 0.125

        self.layers = []
        self.synapsesL0L1 = {}
        self.synapsesL1L2 = {}

        self.expected = 0.000
        self.MSE = 0.000
        self.delta = 0.000
        self.deltagradientL2 = 0.000

        self.build_all_layers_and_neurons()
        self.build_all_synapses()



    def build_all_layers_and_neurons(self):
        self.layers.append(Layer())
        self.layers[0].add_neuron_to_layer(Neuron(0, (100, 225), 0.000, 0.000))
        self.layers[0].add_neuron_to_layer(Neuron(1, (100, 325), 0.000, 0.000))
        #we'll add bias neuron here but no calcs on it
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
        for i in range(0, 2):
            nl0 = self.get_neuron(0, i)
            for j in range(0, 4):
                nl1 = self.get_neuron(1, j)
                self.synapsesL0L1[i, j] = Synapse(self.initial_weight_vals, nl0, nl1)

        setattr(self.synapsesL0L1[0, 0], 'weight', 0.5)

        # synapses between l1 and l2
        for i in range(0, 4):
            nl1 = self.get_neuron(1, i)
            for j in range(0, 1):
                nl2 = self.get_neuron(2, j)
                self.synapsesL1L2[i, j] = Synapse(self.initial_weight_vals, nl1, nl2)

    def get_neuron(self, layerIndex, neuronIndex):
        return self.layers[layerIndex].get_neuron_from_layer(neuronIndex)

    def get_coordinates(self, Neuron):
        return getattr(Neuron, 'coordinatesMidpoint')

    def set_input(self, Neuron, inputn):
        setattr(Neuron, 'inputn', inputn)

    def get_input(self, Neuron):
        return float(getattr(Neuron, 'inputn'))

    def set_output(self, Neuron, output):
        setattr(Neuron, 'output', output)

    def get_output(self, Neuron):
        return float(getattr(Neuron, 'output'))

    def get_weight(self, layerIndex, neuronIndex1, neuronIndex2):
        if layerIndex == 0:
            return getattr(self.synapsesL0L1[neuronIndex1, neuronIndex2], 'weight')
        else:
            return getattr(self.synapsesL1L2[neuronIndex1, neuronIndex2], 'weight')

    def load_inputs(self):
        inputs = self.m_training_data.get_next_inputs()

        self.set_input(self.get_neuron(0, 0), inputs[0])
        self.set_input(self.get_neuron(0, 1), inputs[1])

        self.set_output(self.get_neuron(0, 0), inputs[0])
        self.set_output(self.get_neuron(0, 1), inputs[1])

        self.expected = float(inputs[2])

    def forward_propL0L1(self):


        for i in range(0, 4):
            sum = 0

            for j in range(0, 2):
                prev_neuron = self.get_neuron(0, j)
                prev_neuron_output = self.get_output(prev_neuron)
                weight = self.get_weight(0, j, i)

                sum += prev_neuron_output * weight

            this_neuron = self.get_neuron(1, i)
            self.set_input(this_neuron, str(sum))

            n_output = self.transfer_function(sum)
            self.set_output(this_neuron, n_output)


        a = self.transfer_function(-3434)
        print(a)

    def forward_propL1L2(self):

        sum = 0

        for i in range(0, 4):


            prev_neuron = self.get_neuron(1, i)
            prev_neuron_output = self.get_output(prev_neuron)
            weight = self.get_weight(1, i, 0)

            sum += prev_neuron_output * weight

            this_neuron = self.get_neuron(2, 0)
            self.set_input(this_neuron, str(sum))

            n_output = self.transfer_function(sum)
            self.set_output(this_neuron, n_output)

    def calculate_MSE_and_deltagradient(self):
        pass

    def back_propL2L1(self):
        pass

    def back_propL1L0(self):
        pass

    def update_weights(self):
        pass

    def transfer_function(self, x):
        #use hyperbolic tan
        return math.tanh(x)



