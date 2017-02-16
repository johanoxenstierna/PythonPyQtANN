
import math
# ffgfg
from PythonPyQtANN.Neuron import Neuron
from PythonPyQtANN.Layer import Layer
from PythonPyQtANN.TrainingData import TrainingData

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

        # training Data
        self.m_training_data = TrainingData()
        
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
        self.recent_avg_error = 0
        self.recent_average_smoothing_factor = 0

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

    def get_weight(self, synapseSet, neuronIndex1, neuronIndex2):
        if synapseSet == 0:
            return getattr(self.synapsesl0l1[neuronIndex1, neuronIndex2], 'weight')
        else:
            return getattr(self.synapsesl1l2[neuronIndex1, neuronIndex2], 'weight')

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

    def init_and_draw_next_inputs(self):
        #this is highly hard-coded

        inputs = self.m_training_data.get_next_inputs()

        self.set_input(self.get_neuron(0,0), inputs[0])
        self.set_input(self.get_neuron(0,1), inputs[1])

        setattr(self.get_neuron(0, 0), 'inputn', inputs[0])
        setattr(self.get_neuron(0, 1), 'inputn', inputs[1])

        setattr(self.get_neuron(0, 0), 'output', inputs[0])
        setattr(self.get_neuron(0, 1), 'output', inputs[1])

        self.expected = float(inputs[2])

        # print(inputs[2] + "  " + self.expected)
        # self.expected = inputs[3]
        # self.expected = 54.3


    # def convertToString(self):
    #     return str(self.layers[0])

    def myNetMethod(self):
        w2 = self.synapsesl0l1[0]

    def get_layer(self, index):
        return self.layers[index]

    def forward_propL1(self):

        # for each neuron in layer 1
        for i in range(0, 4):

            sum = 0

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
            
    def forward_propL2(self):

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

        # save above as example. first started doing rounding here but then realized rounding should be donw in viewer exclusively.

    def back_propL1(self):

        self.error = 0.0
        output_neuron = self.get_neuron(2, 0)
        outputVal = self.get_output(output_neuron)
        delta = self.expected - outputVal

        self.error = delta * delta

        # get average error
        self.error /= 1

        # get squared error
        self.error = math.sqrt(self.error)

        #calculate hidden layer gradients
        for i in range(0, 4):
            neuron = self.get_neuron(1, i)

            # calculate hidden gradients
            # calculate derivatives of weights
            # get neuron in last layer
            my_gradient = delta * self.get_weight(1, i, 0)

            setattr(self.get_neuron(1, i), 'gradient', my_gradient)
            print(my_gradient)


    def back_propL2(self):
        print("adsdsafdsa")

        for i in range(0, 2):

            sum = 0.0

            for j in range(0, 4):

                output_neuron = self.get_neuron(1, j)
                previous_gradient = getattr(output_neuron, 'gradient')
                current_weight = self.get_weight(0, i, j)

                my_gradient = previous_gradient * current_weight

                sum += my_gradient

            setattr(self.get_neuron(0, i), 'gradient', sum)




        #
        #
        # # get average error
        # self.error /= 1
        #
        # # get squared error
        # self.error = math.sqrt(self.error)
        #
        # #calculate hidden layer gradients
        # for i in range(0, 4):
        #     neuron = self.get_neuron(1, i)
        #
        #     # calculate hidden gradients
        #     # calculate derivatives of weights
        #     # get neuron in last layer
        #     my_gradient = delta * self.get_weight(1, i, 0)
        #
        #     setattr(self.get_neuron(1, i), 'gradient', my_gradient)
        #     print(my_gradient)


    def calc_output_gradients(self):
        pass


    
    # def back_propL0(self):
            
    def transfer_function(self, x):
        #use hyperbolic tan
        return math.tanh(x)







