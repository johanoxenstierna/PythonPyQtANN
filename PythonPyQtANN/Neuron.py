class Neuron:
    #coordinates is the only thing that is GUI related here
    def __init__(self, index, coordinates, inputn, output):
        super().__init__()

        self.neuronIndex = index
        self.coordinatesMidpoint = coordinates
        self.inputn = inputn
        self.output = output

    def __repr__(self):
        return "Neuron(i:{},xy:'{}',inp:{},out:'{}')".format(self.neuronIndex, self.coordinatesMidpoint, self.inputn, self.output)


