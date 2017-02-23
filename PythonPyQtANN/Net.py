
import math


# ffgfg

from PythonPyQtANN.Neuron import Neuron
from PythonPyQtANN.Layer import Layer
from PythonPyQtANN.TrainingData import TrainingData

class Synapse(object):
    def __init__(self, weight, neuron_left, neuron_right):
        self.weight = weight
        self.deltaweight = 0.000
        self.DOW = 0.000
        self.neuron_left = neuron_left
        self.neuron_right = neuron_right

    def __repr__(self):
        return "Synapse(weight:{}, \n" \
               "    left:'{}',\n" \
               "    right:'{}'\n)".format(self.weight, self.neuron_left, self.neuron_right)




class Net:
    def __init__(self):
        super().__init__()
        #transfer function VS normalization
        #backp
        #
        # training Data
        self.m_training_data = TrainingData()
        
        #initital variables
        self.inititial_weight_vals = 0.1
        self.learning_rate = 0.1
        self.momentum = 0.0


        #MUTABLE : CHANGE CONTENT OF OBJECT WITHOUT CHANGING IDENTITY
        self.layers = []
        self.synapsesl0l1 = {}
        self.synapsesl1l2 = {}

        self.sumDOWL2L1 = 0.000
        self.sumDOWL1L0 = 0.000


        self.build_all_layers_and_neurons()
        self.build_all_synapses()

        self.expected = 0.000
        self.error = 0.000
        self.delta = 0.000
        self.deltagradientL2 = 0.000
        self.deltagradientsL1 = []
        self.recent_avg_error = 0
        self.recent_average_smoothing_factor = 0

        # a = self.transfer_function_derivative(1)
        # print("asfdsadf: " + str(a))




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

        setattr(self.synapsesl0l1[0, 0], 'weight', 0.5)



        # synapses between l1 and l2
        for i in range(0, 4):
            nl1 = self.get_neuron(1, i)
            for j in range(0, 1):
                nl2 = self.get_neuron(2, j)
                self.synapsesl1l2[i, j] = Synapse(self.inititial_weight_vals, nl1, nl2)

        # setattr(self.synapsesl1l2[0, 0], 'weight', 0.5)

    def get_weight(self, synapseSet, neuronIndex1, neuronIndex2):
        if synapseSet == 0:
            return getattr(self.synapsesl0l1[neuronIndex1, neuronIndex2], 'weight')
        else:
            return getattr(self.synapsesl1l2[neuronIndex1, neuronIndex2], 'weight')

    def get_deltaweight(self, synapseSet, neuronIndex1, neuronIndex2):
        if synapseSet == 0:
            return getattr(self.synapsesl0l1[neuronIndex1, neuronIndex2], 'deltaweight')
        else:
            return getattr(self.synapsesl1l2[neuronIndex1, neuronIndex2], 'deltaweight')

    def get_neuron(self, layerIndex, neuronIndex):
        return self.layers[layerIndex].get_neuron_from_layer(neuronIndex)

        # self.layers[index]

        # print("Net, getting: " + index)
        # self.get_neuron_from_layer(index)

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

    def get_gradient(self, Neuron):
        return float(getattr(Neuron, 'gradient'))

    def load_inputs(self):
        #this is highly hard-coded

        inputs = self.m_training_data.get_next_inputs()

        self.set_input(self.get_neuron(0,0), inputs[0])
        self.set_input(self.get_neuron(0,1), inputs[1])

        setattr(self.get_neuron(0, 0), 'inputn', inputs[0])
        setattr(self.get_neuron(0, 1), 'inputn', inputs[1])

        setattr(self.get_neuron(0, 0), 'output', inputs[0])
        setattr(self.get_neuron(0, 1), 'output', inputs[1])

        self.expected = float(inputs[2])

    def get_layer(self, index):
        return self.layers[index]

    def forward_propL0L1(self):

        # for each neuron in layer 1
        for i in range(0, 4):

            sum = 0.0

            # for each neuron in layer 0
            for j in range(0, 2):
                prev_neuron = self.get_neuron(0, j)
                prev_neuron_output = self.get_output(prev_neuron)
                weight = self.get_weight(0, j, i)

                sum += prev_neuron_output * weight

            this_neuron = self.get_neuron(1, i)
            self.set_input(this_neuron, str(sum))

            n_output = self.transfer_function(sum)
            self.set_output(this_neuron, n_output)

            # print(self.get_weight(0, 0, 0))



            # save above as example. first started doing rounding here but then realized rounding should be donw in viewer exclusively.
            
    def forward_propL1L2(self):

        sum = 0.0

        for i in range(0, 4):
            prev_neuron = self.get_neuron(1, i)
            prev_neuron_output = self.get_output(prev_neuron)
            weight = self.get_weight(1, i, 0)

            sum += prev_neuron_output * weight


        this_neuron = self.get_neuron(2, 0)
        self.set_input(this_neuron, str(sum))

        n_output = self.transfer_function(sum)
        self.set_output(this_neuron, n_output)

        # save above as example. first started doing rounding here but then realized rounding should be donw in viewer exclusively.

    def calculate_MSE(self):
        self.delta = self.expected - self.get_output(self.get_neuron(2, 0))

        # calculate MSE
        self.error = self.delta * self.delta

        # get average error
        self.error /= 1

        # get squared error
        self.error = math.sqrt(self.error)

        #calculate deltagradient
        self.deltagradientL2 = self.delta * self.transfer_function_derivative(self.get_output(self.get_neuron(2, 0)))

    def back_propL2L1(self):

        # for each neuron in layer 1
        for i in range(0, 4):

            # calculate sum DOW derivatives of weights
            # sumDOW = 0.0
            # for j in range(0, 4):
            #     # sumDOW += self.get_weight(1, i, 0) * self.deltagradientL2
            #


            #the dow (a weighting factor) is then multiplied to this neurons output val
            # this_output = self.get_output(self.get_neuron(2, 0))
            # new_deltaweight = sumDOW * self.transfer_function_derivative(this_output)
            # print("transfer_function_derivative(this_output): " + str(self.transfer_function_derivative(this_output)))
            # setattr(self.synapsesl1l2[i, 0], 'deltaweight', new_deltaweight)

            g = self.get_output(self.get_neuron(1, i)) * self.deltagradientL2

            setattr(self.synapsesl1l2[i, 0], 'deltaweight', g)



    def back_propL1L0(self):
        # for each neuron in layer 2
        for i in range(0, 2):




    def calc_output_gradients(self):
        pass

    # def back_propL0(self):
            
    def transfer_function(self, x):
        return 1 / (1 + math.pow(math.e,  -x))


    def update_weights(self):
        # update first set

        # 0.1 *

        for i in range(0, 2):
            for j in range(0, 4):
                neuron = self.get_neuron(0, i)
                old_gradient = self.get_gradient(neuron)
                new_gradient = self.learning_rate * self.get_output(neuron) * self.get_gradient(neuron) + \
                    self.momentum * old_gradient

                print("output: " + str(self.get_output(neuron)))

                # setattr(neuron, 'gradient', new_gradient)
                weight = getattr(self.synapsesl0l1[i, j], 'weight')
                weight += new_gradient
                setattr(self.synapsesl0l1[i, j], 'weight', weight)

    def transfer_function_derivative(self, y):
        return y * (1 - y)







