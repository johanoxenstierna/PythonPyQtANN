
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
        self.update_type = "deltaweight"

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
        self.data_row_counter = 0
        self.end_of_data_press_again = False

    def build_all_layers_and_neurons(self):
        self.layers.append(Layer())
        self.layers[0].add_neuron_to_layer(Neuron(0, (100, 150), 0.000, 0.000))
        self.layers[0].add_neuron_to_layer(Neuron(1, (100, 250), 0.000, 0.000))
        self.layers[0].add_neuron_to_layer(Neuron(2, (100, 350), 0.000, 0.000))
        self.layers[0].add_neuron_to_layer(Neuron(3, (100, 450), 0.000, 1.0)) #XX1

        self.layers.append(Layer())
        self.layers[1].add_neuron_to_layer(Neuron(0, (400, 150), 0.000, 0.000))#XX1
        self.layers[1].add_neuron_to_layer(Neuron(1, (400, 250), 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(4, (400, 450), 0.000, 1.0))

        self.layers.append(Layer())
        self.layers[2].add_neuron_to_layer(Neuron(0, (700, 275), 0.000, 0.000))
        # self.layers[2].add_neuron_to_layer(Neuron(1, (700, 375), 0.000, 0.000))

    def build_all_synapses(self): # xx2 major changes

        self.synapsesL0L1[0, 0] = Synapse(self.initial_weight_vals, self.get_neuron(0, 0), self.get_neuron(1, 0))
        self.synapsesL0L1[1, 0] = Synapse(self.initial_weight_vals, self.get_neuron(0, 1), self.get_neuron(1, 0))

        self.synapsesL0L1[1, 1] = Synapse(self.initial_weight_vals, self.get_neuron(0, 1), self.get_neuron(1, 1))
        self.synapsesL0L1[2, 1] = Synapse(self.initial_weight_vals, self.get_neuron(0, 2), self.get_neuron(1, 1))

        #bias
        self.synapsesL0L1[3, 0] = Synapse(self.initial_weight_vals, self.get_neuron(0, 3), self.get_neuron(1, 0))
        self.synapsesL0L1[3, 1] = Synapse(self.initial_weight_vals, self.get_neuron(0, 3), self.get_neuron(1, 1))


        # synapses between l1 and l2
        for i in range(0, 3):
            nl1 = self.get_neuron(1, i)
            nl2 = self.get_neuron(2, 0)
            self.synapsesL1L2[i, 0] = Synapse(self.initial_weight_vals, nl1, nl2)

    def initialize_weights(self):
        # # xx3 WORKS
        # # L0 INPUTS THEY HAVE THE SAME WEIGHTS!
        # setattr(self.synapsesL0L1[0, 0], 'weight', 10.0)
        # setattr(self.synapsesL0L1[1, 0], 'weight', 7.0)
        #
        # setattr(self.synapsesL0L1[1, 1], 'weight', 10.0)
        # setattr(self.synapsesL0L1[2, 1], 'weight', 7.0) # SAME WEIGHTS AS ABOVE
        #
        # setattr(self.synapsesL0L1[3, 0], 'weight', 2.0) #bias
        # setattr(self.synapsesL0L1[3, 1], 'weight', -10.0)
        #
        # #L1
        # setattr(self.synapsesL1L2[0, 0], 'weight', -3.0)
        # setattr(self.synapsesL1L2[1, 0], 'weight', 9.0)
        # setattr(self.synapsesL1L2[2, 0], 'weight', -3.0)


        # # xx3 RANDOM
        # # L0 INPUTS THEY HAVE THE SAME WEIGHTS!
        # w1 = random.uniform(-1.9, 1.9)
        # w2 = random.uniform(-1.9, 1.9)
        #
        # setattr(self.synapsesL0L1[0, 0], 'weight', w1)
        # setattr(self.synapsesL0L1[1, 0], 'weight', w2)
        #
        # setattr(self.synapsesL0L1[1, 1], 'weight', w1)
        # setattr(self.synapsesL0L1[2, 1], 'weight', w2)  # SAME WEIGHTS AS ABOVE
        #
        # setattr(self.synapsesL0L1[3, 0], 'weight', random.uniform(-1.9, 1.9))  # bias
        # setattr(self.synapsesL0L1[3, 1], 'weight', random.uniform(-1.9, 1.9))
        #
        # # L1
        # setattr(self.synapsesL1L2[0, 0], 'weight', random.uniform(-1.9, 1.9))
        # setattr(self.synapsesL1L2[1, 0], 'weight', random.uniform(-1.9, 1.9))
        # setattr(self.synapsesL1L2[2, 0], 'weight', random.uniform(-1.9, 1.9))

        # xx3
        # L0 INPUTS THEY HAVE THE SAME WEIGHTS!
        setattr(self.synapsesL0L1[0, 0], 'weight', 1.0)
        setattr(self.synapsesL0L1[1, 0], 'weight', 1.0)

        setattr(self.synapsesL0L1[1, 1], 'weight', 1.0)
        setattr(self.synapsesL0L1[2, 1], 'weight', 1.0) # SAME WEIGHTS AS ABOVE

        setattr(self.synapsesL0L1[3, 0], 'weight', 1.0) #bias
        setattr(self.synapsesL0L1[3, 1], 'weight', 1.0)

        #L1
        setattr(self.synapsesL1L2[0, 0], 'weight', 1.0)
        setattr(self.synapsesL1L2[1, 0], 'weight', 1.0)
        setattr(self.synapsesL1L2[2, 0], 'weight', 1.0)

        # # L0 INPUTS THEY HAVE THE SAME WEIGHTS!
        # setattr(self.synapsesL0L1[0, 0], 'weight', -0.5)
        # setattr(self.synapsesL0L1[1, 0], 'weight', 11.0)
        #
        # setattr(self.synapsesL0L1[1, 1], 'weight', -0.5)
        # setattr(self.synapsesL0L1[2, 1], 'weight', 11.0) # SAME WEIGHTS AS ABOVE
        #
        # setattr(self.synapsesL0L1[3, 0], 'weight', -10.0) #bias
        # setattr(self.synapsesL0L1[3, 1], 'weight', 0.1)
        #
        # #L1
        # setattr(self.synapsesL1L2[0, 0], 'weight', 11.0)
        # setattr(self.synapsesL1L2[1, 0], 'weight', -0.5)
        # setattr(self.synapsesL1L2[2, 0], 'weight', -5.0)

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
        if inputs == [5, 5, 5, 5]:
            self.data_row_counter = 0 #xx9
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
            # xx8
            self.data_row_counter += 1 #a new datarow is going to be processed
            self.set_input(self.get_neuron(0, 0), inputs[0])
            self.set_input(self.get_neuron(0, 1), inputs[1])
            self.set_input(self.get_neuron(0, 2), inputs[2])

            self.set_output(self.get_neuron(0, 0), inputs[0])
            self.set_output(self.get_neuron(0, 1), inputs[1])
            self.set_output(self.get_neuron(0, 2), inputs[2])

            self.expected = float(inputs[3])

            self.end_of_data_press_again = False

            return True

    def forward_propL0L1(self, filter_index):

        if filter_index == 0:
            prev_neuron1_output = self.get_output(self.get_neuron(0, 0))
            prev_neuron2_output = self.get_output(self.get_neuron(0, 1))
            bias_output = self.get_output(self.get_neuron(0, 3))

            weight1 = self.synapsesL0L1[0, 0].weight
            weight2 = self.synapsesL0L1[1, 0].weight
            weight3bias = self.synapsesL0L1[3, 0].weight

            sum = prev_neuron1_output * weight1 + prev_neuron2_output * weight2 + bias_output * weight3bias
            self.set_input(self.get_neuron(1, 0), str(sum))

            n_output = self.transfer_function(sum)
            self.set_output(self.get_neuron(1, 0), n_output)

        else:
            prev_neuron1_output = self.get_output(self.get_neuron(0, 1))
            prev_neuron2_output = self.get_output(self.get_neuron(0, 2))
            bias_output = self.get_output(self.get_neuron(0, 3))

            weight1 = self.synapsesL0L1[1, 1].weight
            weight2 = self.synapsesL0L1[2, 1].weight
            weight3bias = self.synapsesL0L1[3, 1].weight

            sum = prev_neuron1_output * weight1 + prev_neuron2_output * weight2 + bias_output * weight3bias
            self.set_input(self.get_neuron(1, 1), str(sum))

            n_output = self.transfer_function(sum)
            self.set_output(self.get_neuron(1, 1), n_output)

    def forward_propL1L2(self):

        sum = 0

        for i in range(0, 3): # L1

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

        self.MSE = pow(self.delta, 2)
        self.epoch_MSEs.append(self.MSE)
        self.deltagradientL2 = self.delta * self.transfer_function_derivative(self.get_output(self.get_neuron(2, 0)))

    def back_propL2L1(self):
        for i in range(0, 2): #L2
            deltaweight = self.get_output(self.get_neuron(1, i)) * self.deltagradientL2
            setattr(self.synapsesL1L2[i, 0], 'deltaweight', deltaweight)

            batch_deltaweight = getattr(self.synapsesL1L2[i, 0], 'batch_deltaweight')
            new_batch_deltaweight = batch_deltaweight * (
                self.data_row_counter - 1) / self.data_row_counter + deltaweight / self.data_row_counter
            setattr(self.synapsesL1L2[i, 0], 'batch_deltaweight', new_batch_deltaweight)

        #bias L1L2
        deltaweight = self.deltagradientL2
        setattr(self.synapsesL1L2[2, 0], 'deltaweight', deltaweight)

        #bias batch L1L2
        #New average = old average * (n-1)/n + new value /n
        batch_deltaweight = getattr(self.synapsesL1L2[2, 0], 'batch_deltaweight')
        new_batch_deltaweight = batch_deltaweight * (
            self.data_row_counter - 1)/self.data_row_counter + deltaweight / self.data_row_counter
        setattr(self.synapsesL1L2[2, 0], 'batch_deltaweight', new_batch_deltaweight)

    def back_propL1L0(self, filter_index):

        for synapse in self.synapsesL0L1:
            if synapse[1] == filter_index: #synapse attached to certain neuron in L1
                neuronL0 = self.get_neuron(0, synapse[0])
                neuronL1 = self.get_neuron(1, synapse[1])

                # the following part requires chronologically correct pressing of buttons
                other_deltaweight = 0
                other_batch_deltaweight = 0

                if filter_index == 1: # the prev_deltaweights are collected from the previous filter location
                    if synapse[0] == 1: # if the first connection is neuron 1 in L0
                        other_deltaweight = self.get_deltaweight(0, 0, 0) # there are several options
                        other_batch_deltaweight = self.get_batch_deltaweight(0, 0, 0)
                    elif synapse[0] == 2:
                        other_deltaweight = self.get_deltaweight(0, 1, 0) #there are several options
                        other_batch_deltaweight = self.get_batch_deltaweight(0, 1, 0)
                    # else: the bias is used and it uses no prev weight
                # else:
                #     other_deltaweight = 0
                #     other_batch_deltaweight = 0

                part1 = self.transfer_function_derivative(self.get_output(neuronL1)) * self.deltagradientL2
                part2 = self.get_weight(1, synapse[1], 0) #this means L1L2, neuron in L1, neuron in L2
                part3 = self.get_output(neuronL0)
                deltaweight = other_deltaweight + part1 * part2 * part3
                setattr(self.synapsesL0L1[synapse[0], synapse[1]], 'deltaweight', deltaweight)

                batch_deltaweight = getattr(self.synapsesL0L1[synapse[0], synapse[1]], 'batch_deltaweight')
                new_batch_deltaweight = other_batch_deltaweight + batch_deltaweight * (
                self.data_row_counter - 1) / self.data_row_counter + deltaweight / self.data_row_counter
                setattr(self.synapsesL0L1[synapse[0], synapse[1]], 'batch_deltaweight', new_batch_deltaweight)

                # if filter index == 1 the filter_index 1 weights should be updated (2 synapses with 2 weights each but they are the same):
                if filter_index == 1: # JESUS F CHRIST
                    if synapse[0] == 1:
                        setattr(self.synapsesL0L1[0, 0], 'deltaweight', deltaweight)
                        setattr(self.synapsesL0L1[0, 0], 'batch_deltaweight', new_batch_deltaweight)
                    elif synapse[0] == 2:
                        setattr(self.synapsesL0L1[1, 0], 'deltaweight', deltaweight)
                        setattr(self.synapsesL0L1[1, 0], 'batch_deltaweight', new_batch_deltaweight)

        #bias
        neuronL1 = self.get_neuron(1, filter_index) # filter_index = neuron in L1
        part1 = self.transfer_function_derivative(self.get_output(neuronL1)) * self.deltagradientL2
        part2 = self.get_weight(1, filter_index, 0)
        deltaweight = part1 * part2 # no part 3 here!
        setattr(self.synapsesL0L1[3, filter_index], 'deltaweight', deltaweight)

        # bias batch
        batch_deltaweight = getattr(self.synapsesL0L1[3, filter_index], 'batch_deltaweight')
        new_batch_deltaweight = batch_deltaweight * (
            self.data_row_counter - 1) / self.data_row_counter + deltaweight / self.data_row_counter
        setattr(self.synapsesL0L1[3, filter_index], 'batch_deltaweight', new_batch_deltaweight)

        aa=4

    def update_weights(self, update_type):
        # update first set
        print("Updating weights")

        for synapse in self.synapsesL0L1:
            weight = self.get_weight(0, synapse[0], synapse[1])
            # either batch or pattern
            parsed_update_type = eval("self.get_" + update_type + "(0, " + str(synapse[0]) + ", " + str(synapse[1]) + ")")
            weight_change = self.learning_rate * parsed_update_type + \
                            self.momentum * weight
            weight += weight_change
            setattr(self.synapsesL0L1[synapse[0], synapse[1]], 'weight', weight)

        for synapse in self.synapsesL1L2:
            weight = self.get_weight(1, synapse[0], synapse[1])
            # either batch or pattern
            parsed_update_type = eval(
                "self.get_" + update_type + "(1, " + str(synapse[0]) + ", " + str(synapse[1]) + ")")
            weight_change = self.learning_rate * parsed_update_type + \
                            self.momentum * weight
            weight += weight_change
            setattr(self.synapsesL1L2[synapse[0], synapse[1]], 'weight', weight)


        aa = 5
        # for i in range(0, 3):
        #     for j in range(0, 4):
        #         weight = self.get_weight(0, i, j)
        #
        #         parsed_update_type = eval("self.get_" + update_type + "(0, " + str(i) + ", " + str(j) + ")")
        #         weight_change = self.learning_rate * parsed_update_type + \
        #             self.momentum * weight
        #         weight += weight_change
        #         setattr(self.synapsesL0L1[i, j], 'weight', weight)
        #
        # # update second set
        # for i in range(0, 5):
        #     weight = self.get_weight(1, i, 0)
        #     parsed_update_type = eval("self.get_" + update_type + "(1, " + str(i) + ", 0)")
        #     weight_change = self.learning_rate * parsed_update_type + \
        #                     self.momentum * weight
        #
        #     weight += weight_change
        #     setattr(self.synapsesL1L2[i, 0], 'weight', weight)

    def transfer_function(self, y):
        #Sigmoid
        return 1 / (1 + math.pow(math.e, -y))

        # # #ReLU
        # return max(0, y)

    def transfer_function_derivative(self, y):
        # Sigmoid
        return 1 / (1 + math.pow(math.e, -y)) * (1 - 1 / (1 + math.pow(math.e, -y)))

        # # ReLU
        # if y <= 0:
        #     return 0
        # else:
        #     return 1

    def run_epoch(self):

        there_is_data = True
        while there_is_data == True:
            there_is_data = self.load_inputs()
            self.end_of_data_press_again = False
            if there_is_data == False:
                break
            else:
                self.forward_propL0L1(0)
                self.forward_propL0L1(1)
                self.forward_propL1L2()
                self.calculate_MSE_and_deltagradient()
                self.back_propL2L1()
                self.back_propL1L0(0)
                self.back_propL1L0(1)

                if self.update_type == "deltaweight":
                    self.update_weights(self.update_type)

            print("one data row finished")

    def epoch_MSEs_and_reset_synapse_batches(self):
        print("reset batches")
        self.batch_MSE = sum(self.epoch_MSEs) / len(self.epoch_MSEs)
        self.epoch_MSEs.clear()
        #reset batch deltaweights

        for synapse in self.synapsesL0L1:
            setattr(self.synapsesL0L1[synapse[0], synapse[1]], 'batch_deltaweight', 0.0)
            setattr(self.synapsesL0L1[synapse[0], synapse[1]], 'deltaweight', 0.0)

        for synapse in self.synapsesL1L2:
            setattr(self.synapsesL1L2[synapse[0], synapse[1]], 'batch_deltaweight', 0.0)
            setattr(self.synapsesL1L2[synapse[0], synapse[1]], 'deltaweight', 0.0)

        # for i in range(0, 3):
        #     for j in range(0, 4):
        #         setattr(self.synapsesL0L1[i, j], 'batch_deltaweight', 0.0)
        # for i in range(0, 4):
        #     setattr(self.synapsesL1L2[i, 0], 'batch_deltaweight', 0.0)


