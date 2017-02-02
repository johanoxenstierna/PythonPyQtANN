
class Neuron:
    #coordinates is the only thing that is GUI related here
    def __init__(self, index, coordinates, inputn, output):
        super().__init__()

        self.neuronIndex = index
        self.coordinatesMidpoint = coordinates
        self.inputn = inputn
        self.output = output

    # def add_input_to_neuron(self, index, an_input):
    #     self.input.append({index, an_input})

    # def set_input(self, inputn):
    #     self.inputn = inputn

    # def get_input(self):
    #     return self.inputn

    # def get_coordinates(self):
    #     return self.coordinatesMidpoint

    # def set_outputs(self, outputs):
    #     self.output = outputs
    #
    # def get_outputs(self):
    #     return self.output

    # def set_output_weight(self, o_weights):
    #     self.o_weights = o_weights
    #
    # def get_output_weights(self):
    #     return self.o_weights

    def __repr__(self):
        return "Neuron(i:{},xy:'{}',inp:{},out:'{}')".format(self.neuronIndex, self.coordinatesMidpoint, self.inputn, self.output)






