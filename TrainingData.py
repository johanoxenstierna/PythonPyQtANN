
class TrainingData:
    def __init__(self):
        super().__init__()
        self.text_file = open("myTrainingDataFile.txt")
        # print(input1 + " " + input2)


    def get_next_inputs(self):
        next_line = self.text_file.readline()
        next_line_list = next_line.split(";")
        return [next_line_list[1], next_line_list[2], next_line_list[4]]


    def end_file_read(self):
        self.text_file.close()

