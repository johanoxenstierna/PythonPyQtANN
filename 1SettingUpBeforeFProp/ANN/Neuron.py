class Neuron:
    def __init__(self, index, coordinates, inputn, output):
        super().__init__()
        self.index = index
        self.coordinatesMidpoint = coordinates
        self.inputn = inputn
        self.output = output

    def __repr__(self):
        return "Neuron(i:{}, xy:'{}', inp:{}, out:'{}'".format(
            self.index, self.coordinatesMidpoint, self.inputn, self.output
        )

