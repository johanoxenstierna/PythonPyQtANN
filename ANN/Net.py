
import math, random
from ANN.TrainingData import TrainingData

from ANN.Layer import Layer
from ANN.Neuron import Neuron
from ANN.TrainingData import TrainingData

class Synapse:
    def __init__(self, weight, neuron_left, neuron_right):
        self.weight = weight
        self.deltaweight = 0.000
        self.batch_deltaweight = 0.000
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

        self.initial_weight_vals = 0.01
        self.learning_rate = 1
        self.momentum = 0
        self.update_type = ""

        self.layers = []
        self.synapsesL0L1 = {}
        self.synapsesL1L2 = {}

        self.expected = 0.000
        self.batch_MSE = 0.0
        self.MSE = 0.000
        self.epoch_MSEs = []
        self.delta = 0.000
        self.deltagradientL2 = 0.000

        self.build_all_layers_and_neurons()
        self.build_all_synapses()
        self.initialize_weights()

        self.epoch = 0
        self.end_of_data_press_again = False


    def build_all_layers_and_neurons(self):
        self.layers.append(Layer())
        self.layers[0].add_neuron_to_layer(Neuron(0, (100, 225), 0.000, 0.000))
        self.layers[0].add_neuron_to_layer(Neuron(1, (100, 325), 0.000, 0.000))
        self.layers[0].add_neuron_to_layer(Neuron(2, (100, 425), 0.000, 1.0))

        self.layers.append(Layer())
        self.layers[1].add_neuron_to_layer(Neuron(0, (400, 125), 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(1, (400, 225), 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(2, (400, 325), 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(3, (400, 425), 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(4, (400, 525), 0.000, 1.0))

        self.layers.append(Layer())
        self.layers[2].add_neuron_to_layer(Neuron(0, (700, 275), 0.000, 0.000))
        self.layers[2].add_neuron_to_layer(Neuron(1, (700, 375), 0.000, 0.000))

    def build_all_synapses(self):
        #synapses between l0 and l1
        for i in range(0, 3):
            nl0 = self.get_neuron(0, i)
            for j in range(0, 4):
                nl1 = self.get_neuron(1, j)
                self.synapsesL0L1[i, j] = Synapse(self.initial_weight_vals, nl0, nl1)

        setattr(self.synapsesL0L1[0, 0], 'weight', 0.2)

        # synapses between l1 and l2
        for i in range(0, 5):
            nl1 = self.get_neuron(1, i)
            for j in range(0, 1):
                nl2 = self.get_neuron(2, j)
                self.synapsesL1L2[i, j] = Synapse(self.initial_weight_vals, nl1, nl2)

    def initialize_weights(self):
        # #L0
        # for i in range(0, 3):
        #     for j in range(0, 4):
        #         my_weight = random.uniform(-2, 2)
        #         setattr(self.synapsesL0L1[i, j], 'weight', my_weight)
        #
        # # L1
        # for i in range(0, 5):
        #     my_weight = random.uniform(-2, 2)
        #     setattr(self.synapsesL1L2[i, 0], 'weight', my_weight)

        #batch
        setattr(self.synapsesL0L1[0, 0], 'weight', -0.2)
        setattr(self.synapsesL0L1[0, 1], 'weight', 1.1)
        setattr(self.synapsesL0L1[0, 2], 'weight', -1.3)
        setattr(self.synapsesL0L1[0, 3], 'weight', 0.1)

        setattr(self.synapsesL0L1[1, 0], 'weight', -1.4)
        setattr(self.synapsesL0L1[1, 1], 'weight', -0.9)
        setattr(self.synapsesL0L1[1, 2], 'weight', 1.0)
        setattr(self.synapsesL0L1[1, 3], 'weight', 0.1)

        setattr(self.synapsesL0L1[2, 0], 'weight', -1.1)
        setattr(self.synapsesL0L1[2, 1], 'weight', 0.2)
        setattr(self.synapsesL0L1[2, 2], 'weight', 0.7)
        setattr(self.synapsesL0L1[2, 3], 'weight', 0.6)

        # setattr(self.synapsesL0L1[2, 0], 'weight', 19)

        setattr(self.synapsesL1L2[0, 0], 'weight', -0.8)
        setattr(self.synapsesL1L2[1, 0], 'weight', -1.7)
        setattr(self.synapsesL1L2[2, 0], 'weight', -0.1)
        setattr(self.synapsesL1L2[3, 0], 'weight', -1.0)
        setattr(self.synapsesL1L2[4, 0], 'weight', 1.2)

        # #working
        # setattr(self.synapsesL0L1[0, 0], 'weight', -1.5)
        # setattr(self.synapsesL0L1[0, 1], 'weight', -0.0)
        # setattr(self.synapsesL0L1[0, 2], 'weight', -1.5)
        # setattr(self.synapsesL0L1[0, 3], 'weight', -1.8)
        #
        # setattr(self.synapsesL0L1[1, 0], 'weight', 1.5)
        # setattr(self.synapsesL0L1[1, 1], 'weight', -0.5)
        # setattr(self.synapsesL0L1[1, 2], 'weight', 1.3)
        # setattr(self.synapsesL0L1[1, 3], 'weight', 1)
        #
        # setattr(self.synapsesL0L1[2, 0], 'weight', 1.6)
        # setattr(self.synapsesL0L1[2, 1], 'weight', -0.5)
        # setattr(self.synapsesL0L1[2, 2], 'weight', -0.4)
        # setattr(self.synapsesL0L1[2, 3], 'weight', -0.3)
        #
        # # setattr(self.synapsesL0L1[2, 0], 'weight', 19)
        #
        # setattr(self.synapsesL1L2[0, 0], 'weight', -0.1)
        # setattr(self.synapsesL1L2[1, 0], 'weight', 1.1)
        # setattr(self.synapsesL1L2[2, 0], 'weight', 1.2)
        # setattr(self.synapsesL1L2[3, 0], 'weight', 1.4)
        # setattr(self.synapsesL1L2[4, 0], 'weight', 1.5)

        # #not working
        # setattr(self.synapsesL0L1[0, 0], 'weight', 1.5)
        # setattr(self.synapsesL0L1[0, 1], 'weight', -0.3)
        # setattr(self.synapsesL0L1[0, 2], 'weight', -0.5)
        # setattr(self.synapsesL0L1[0, 3], 'weight', 1.8)
        #
        # setattr(self.synapsesL0L1[1, 0], 'weight', 1.5)
        # setattr(self.synapsesL0L1[1, 1], 'weight', -1.1)
        # setattr(self.synapsesL0L1[1, 2], 'weight', -1.4)
        # setattr(self.synapsesL0L1[1, 3], 'weight', -0.8)
        #
        # setattr(self.synapsesL0L1[2, 0], 'weight', -0.7)
        # setattr(self.synapsesL0L1[2, 1], 'weight', 0.5)
        # setattr(self.synapsesL0L1[2, 2], 'weight', -0.9)
        # setattr(self.synapsesL0L1[2, 3], 'weight', -0.1)
        #
        # # setattr(self.synapsesL0L1[2, 0], 'weight', 19)
        #
        # setattr(self.synapsesL1L2[0, 0], 'weight', -0.1)
        # setattr(self.synapsesL1L2[1, 0], 'weight', 1.1)
        # setattr(self.synapsesL1L2[2, 0], 'weight', -0.2)
        # setattr(self.synapsesL1L2[3, 0], 'weight', 0.4)
        # setattr(self.synapsesL1L2[4, 0], 'weight', -1.5)

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

    def get_deltaweight(self, layerIndex, neuronIndex1, neuronIndex2):
        if layerIndex == 0:
            return getattr(self.synapsesL0L1[neuronIndex1, neuronIndex2], 'deltaweight')
        else:
            return getattr(self.synapsesL1L2[neuronIndex1, neuronIndex2], 'deltaweight')

    def get_batch_deltaweight(self, layerIndex, neuronIndex1, neuronIndex2):
        if layerIndex == 0:
            return getattr(self.synapsesL0L1[neuronIndex1, neuronIndex2], 'batch_deltaweight')
        else:
            return getattr(self.synapsesL1L2[neuronIndex1, neuronIndex2], 'batch_deltaweight')

    def load_inputs(self):
        inputs = self.m_training_data.get_next_inputs()
        if inputs == [0, 0, 0]:
            print("Epoch done")
            self.epoch += 1
            self.end_of_data_press_again = True
            self.m_training_data.move_to_top_of_file()
            print(self.epoch_MSEs)
            if self.update_type == "batch_deltaweight":
                self.update_weights("batch_deltaweight")
            self.epoch_MSEs_and_reset_synapse_batches()
            return False
        else:

            self.set_input(self.get_neuron(0, 0), inputs[0])
            self.set_input(self.get_neuron(0, 1), inputs[1])

            self.set_output(self.get_neuron(0, 0), inputs[0])
            self.set_output(self.get_neuron(0, 1), inputs[1])

            self.expected = float(inputs[2])

            self.end_of_data_press_again = False

            return True

    def forward_propL0L1(self):

        for i in range(0, 4):
            sum = 0

            for j in range(0, 3):
                prev_neuron = self.get_neuron(0, j)
                prev_neuron_output = self.get_output(prev_neuron)
                weight = self.get_weight(0, j, i)

                sum += prev_neuron_output * weight

            this_neuron = self.get_neuron(1, i)
            self.set_input(this_neuron, str(sum))

            n_output = self.transfer_function(sum)
            self.set_output(this_neuron, n_output)

    def forward_propL1L2(self):

        sum = 0

        for i in range(0, 5):

            prev_neuron = self.get_neuron(1, i)
            prev_neuron_output = self.get_output(prev_neuron)
            weight = self.get_weight(1, i, 0)

            sum += prev_neuron_output * weight

            this_neuron = self.get_neuron(2, 0)
            self.set_input(this_neuron, str(sum))

            n_output = self.transfer_function(sum)
            self.set_output(this_neuron, n_output)

    def calculate_MSE_and_deltagradient(self):
        self.delta = self.expected - self.get_output(self.get_neuron(2, 0))

        self.MSE = 0.5 * pow(self.delta, 2)
        self.epoch_MSEs.append(self.MSE)
        self.deltagradientL2 = self.delta * self.transfer_function_derivative(self.get_output(self.get_neuron(2, 0)))

    def back_propL2L1(self):
        for i in range(0, 4):
            deltaweight = self.get_output(self.get_neuron(1, i)) * self.deltagradientL2
            setattr(self.synapsesL1L2[i, 0], 'deltaweight', deltaweight)

            batch_deltaweight = getattr(self.synapsesL1L2[i, 0], 'batch_deltaweight')
            new_batch_deltaweight = deltaweight + batch_deltaweight
            setattr(self.synapsesL1L2[i, 0], 'batch_deltaweight', new_batch_deltaweight)

        #bias
        deltaweight = self.deltagradientL2
        setattr(self.synapsesL1L2[4, 0], 'deltaweight', deltaweight)

        #bias batch
        batch_deltaweight = getattr(self.synapsesL1L2[4, 0], 'batch_deltaweight')
        new_batch_deltaweight = deltaweight + batch_deltaweight
        setattr(self.synapsesL1L2[4, 0], 'batch_deltaweight', new_batch_deltaweight)

    def back_propL1L0(self):
        for i in range(0, 2):
            for j in range(0, 4):
                neuronL1 = self.get_neuron(1, j)
                neuronL0 = self.get_neuron(0, i)
                part1 = self.transfer_function_derivative(self.get_output(neuronL1)) * self.deltagradientL2
                part2 = self.get_weight(1, j, 0)
                part3 = self.get_output(neuronL0)
                deltaweight = part1 * part2 * part3
                setattr(self.synapsesL0L1[i, j], 'deltaweight', deltaweight)

                batch_deltaweight = getattr(self.synapsesL0L1[i, j], 'batch_deltaweight')
                new_batch_deltaweight = deltaweight + batch_deltaweight
                setattr(self.synapsesL0L1[i, j], 'batch_deltaweight', new_batch_deltaweight)
        #bias
        for i in range(0, 4):
            neuronL1 = self.get_neuron(1, i)
            part1 = self.transfer_function_derivative(self.get_output(neuronL1)) * self.deltagradientL2
            part2 = self.get_weight(1, i, 0)
            deltaweight = part1 * part2
            setattr(self.synapsesL0L1[2, i], 'deltaweight', deltaweight)

            # bias batch
            batch_deltaweight = getattr(self.synapsesL0L1[2, i], 'batch_deltaweight')
            new_batch_deltaweight = deltaweight + batch_deltaweight
            setattr(self.synapsesL0L1[2, i], 'batch_deltaweight', new_batch_deltaweight)

    def update_weights(self, update_type):
        # update first set
        print("Updating weights")
        for i in range(0, 3):
            for j in range(0, 4):
                weight = self.get_weight(0, i, j)

                parsed_update_type = eval("self.get_" + update_type + "(0, " + str(i) + ", " + str(j) + ")")
                weight_change = self.learning_rate * parsed_update_type + \
                    self.momentum * weight
                weight += weight_change
                setattr(self.synapsesL0L1[i, j], 'weight', weight)

        # update second set
        for i in range(0, 5):
            weight = self.get_weight(1, i, 0)
            parsed_update_type = eval("self.get_" + update_type + "(1, " + str(i) + ", 0)")
            weight_change = self.learning_rate * parsed_update_type + \
                            self.momentum * weight

            weight += weight_change
            setattr(self.synapsesL1L2[i, 0], 'weight', weight)

    def transfer_function(self, y):
        return 1 / (1 + math.pow(math.e, -y))

    def transfer_function_derivative(self, y):
        return 1 / (1 + math.pow(math.e, -y)) * (1 - 1 / (1 + math.pow(math.e, -y)))

    def run_epoch(self):

        there_is_data = True
        while there_is_data == True:
            there_is_data = self.load_inputs()
            self.end_of_data_press_again = False
            if there_is_data == False:
                break
            else:
                self.forward_propL0L1()
                self.forward_propL1L2()
                self.calculate_MSE_and_deltagradient()
                self.back_propL2L1()
                self.back_propL1L0()

                if self.update_type == "deltaweight":
                    self.update_weights(self.update_type)

            print("one data row finished")

    def epoch_MSEs_and_reset_synapse_batches(self):
        print("reset batches")
        self.batch_MSE = sum(self.epoch_MSEs) / len(self.epoch_MSEs)
        self.epoch_MSEs.clear()
        #reset batch deltaweights
        for i in range(0, 3):
            for j in range(0, 4):
                setattr(self.synapsesL0L1[i, j], 'batch_deltaweight', 0.0)
        for i in range(0, 4):
            setattr(self.synapsesL1L2[i, 0], 'batch_deltaweight', 0.0)


