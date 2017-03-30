
import sys, random, math

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from ANN.Net import Net
from ANN.TrainingData import TrainingData

class Controller(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(700, 100, 900, 700)

        self.l0inputs = []
        self.l0outputs = []
        self.l0wsum = QLineEdit(self)

        self.l0l1weights = {}

        self.l1inputs = []
        self.l1outputs = []
        self.l1wsum = QLineEdit(self)

        self.l1l2weights = {}

        self.l2inputs = []
        self.l2outputs = []

        self.expected_v = QLineEdit(self)
        self.error_v = QLineEdit(self)

        self.initialize_and_draw_lineEdits()

        self.load_inputs_button = QPushButton(self)
        self.load_inputs_button.clicked.connect(self.init_and_draw_next_inputs)

        self.initialize_buttons()


        self.show()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_neurons(qp)
        self.draw_synapses_and_labels(qp)
        self.update_lineEdits()

        qp.end()

    def draw_neurons(self, qp):
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        #draw first layer
        for i in range(0, 3):
            c = m_net.get_coordinates(m_net.get_neuron(0, i))
            qp.drawEllipse(c[0] - 25, c[1] - 25, 50, 50)

        # draw second layer
        for i in range(0, 5):
            c = m_net.get_coordinates(m_net.get_neuron(1, i))
            qp.drawEllipse(c[0] - 25, c[1] - 25, 50, 50)

        # draw third layer
        for i in range(0, 2):
            c = m_net.get_coordinates(m_net.get_neuron(2, i))
            qp.drawEllipse(c[0] - 25, c[1] - 25, 50, 50)

    def draw_synapses_and_labels(self, qp):
        pen = QPen(Qt.black, 1, Qt.DashLine)
        qp.setPen(pen)

        qp.drawLine(175, 225, 210, 200)
        qp.drawLine(175, 225, 210, 220)
        qp.drawLine(175, 225, 210, 240)
        qp.drawLine(175, 225, 210, 260)

        qp.drawLine(175, 325, 210, 295)
        qp.drawLine(175, 325, 210, 315)
        qp.drawLine(175, 325, 210, 335)
        qp.drawLine(175, 325, 210, 355)

        qp.drawLine(260, 200, 310, 125)
        qp.drawLine(260, 220, 310, 225)
        qp.drawLine(260, 240, 310, 325)
        qp.drawLine(260, 260, 310, 425)

        qp.drawLine(260, 300, 310, 125)
        qp.drawLine(260, 320, 310, 225)
        qp.drawLine(260, 340, 310, 325)
        qp.drawLine(260, 360, 310, 425)

        qp.drawLine(460, 125, 510, 125)
        qp.drawLine(460, 225, 510, 225)
        qp.drawLine(460, 325, 510, 325)
        qp.drawLine(460, 425, 510, 425)

        qp.drawLine(555, 125, 625, 275)
        qp.drawLine(555, 225, 625, 275)
        qp.drawLine(555, 325, 625, 275)
        qp.drawLine(555, 425, 625, 275)

        qp.drawText(125, 210, "out")
        qp.drawText(210, 180, "w's")
        qp.drawText(210, 140, "sum w")

        qp.drawText(425, 110, "out")
        qp.drawText(510, 110, "w's")
        qp.drawText(510, 70, "sum w")

        qp.drawText(730, 260, "out")
        qp.drawText(780, 260, "expc")
        qp.drawText(845, 260, "err")

    def initialize_buttons(self):
        self.load_inputs_button.setGeometry(10, 500, 100, 60)
        self.load_inputs_button.setText("Load inp")

    def initialize_and_draw_lineEdits(self):
        #first layer
        for i in range(0, 2):
            c = m_net.get_coordinates(m_net.get_neuron(0, i))

            #inputs
            my_line_edit_in = QLineEdit(self)
            my_line_edit_in.setGeometry(c[0] - 90, c[1] - 10, 60, 20)
            self.l0inputs.append([i, my_line_edit_in])

            #outputs
            my_line_edit_out = QLineEdit(self)
            my_line_edit_out.setGeometry(c[0] + 25, c[1] - 10, 50, 20)
            self.l0outputs.append([i, my_line_edit_out])

            #weights
            for j in range(0, 4):
                my_line_edit_w = QLineEdit(self)
                my_line_edit_w.setGeometry(c[0] + 110, c[1] - 40 + (22 * j), 50, 20)
                self.l0l1weights[i, j] = my_line_edit_w

        c = m_net.get_coordinates(m_net.get_neuron(0, 0))
        self.l0wsum.setGeometry(c[0] + 110, c[1] - 80, 50, 20)


        #second layer
        for i in range(0, 4):
            c = m_net.get_coordinates(m_net.get_neuron(1, i))

            #inputs
            my_line_edit_in = QLineEdit(self)
            my_line_edit_in.setGeometry(c[0] - 90, c[1] - 10, 60, 20)
            self.l1inputs.append([i, my_line_edit_in])

            #outputs
            my_line_edit_out = QLineEdit(self)
            my_line_edit_out.setGeometry(c[0] + 25, c[1] - 10, 50, 20)
            self.l1outputs.append([i, my_line_edit_out])

            #weights
            my_line_edit_w = QLineEdit(self)
            my_line_edit_w.setGeometry(c[0] + 110, c[1] - 10, 50, 20)
            self.l1l2weights[i, 0] = my_line_edit_w


        c = m_net.get_coordinates(m_net.get_neuron(1, 0))
        self.l1wsum = QLineEdit(self)
        self.l1wsum.setGeometry(c[0] + 110, c[1] - 50, 50, 20)



        #third layer
        for i in range(0, 1):
            c = m_net.get_coordinates(m_net.get_neuron(2, i))

            #inputs
            my_line_edit_in = QLineEdit(self)
            my_line_edit_in.setGeometry(c[0] - 90, c[1] - 10, 60, 20)
            self.l2inputs.append([i, my_line_edit_in])

            #outputs
            my_line_edit_out = QLineEdit(self)
            my_line_edit_out.setGeometry(c[0] + 25, c[1] - 10, 50, 20)
            self.l2outputs.append([i, my_line_edit_out])


        #draw expected output and error
        self.expected_v.setGeometry(780, 265, 50, 20)
        self.error_v.setGeometry(840, 265, 50, 20)

    def update_lineEdits(self):

        #update first layer
        synapses = getattr(m_net, 'synapsesl0l1')

        for i in range(0, 2):
            #input
            input = m_net.get_input(m_net.get_neuron(0, i))
            QLineEdit.setText(self.l0inputs[i][1], str(round(input, 3)))

            #outputs
            output = m_net.get_output(m_net.get_neuron(0, i))
            QLineEdit.setText(self.l0outputs[i][1], str(round(output, 3)))

            #weights
            for j in range(0, 4):
                weight = getattr(synapses[i, 0], 'weight')
                QLineEdit.setText(self.l0l1weights[i, j], str(weight))


        # update second layer
        synapses = getattr(m_net, 'synapsesl1l2')

        for i in range(0, 4):

            # input
            input = m_net.get_input(m_net.get_neuron(1, i))
            QLineEdit.setText(self.l1inputs[i][1], str(round(input, 3)))

            # outputs
            output = m_net.get_output(m_net.get_neuron(1, i))
            QLineEdit.setText(self.l1outputs[i][1], str(round(output, 3)))

            # weights
            weight = getattr(synapses[i, 0], 'weight')
            QLineEdit.setText(self.l1l2weights[i, 0], str(weight))


        # update third layer

        for i in range(0, 1):

            # input
            input = m_net.get_input(m_net.get_neuron(2, i))
            QLineEdit.setText(self.l2inputs[i][1], str(round(input, 3)))

            # outputs
            output = m_net.get_output(m_net.get_neuron(2, i))
            QLineEdit.setText(self.l2outputs[i][1], str(round(output, 3)))

    def init_and_draw_next_inputs(self):
        inputs = m_training_data.get_next_inputs()

        m_net.set_input(m_net.get_neuron(0, 0), inputs[0])
        m_net.set_input(m_net.get_neuron(0, 1), inputs[1])

        m_net.set_output(m_net.get_neuron(0, 0), inputs[0])
        m_net.set_output(m_net.get_neuron(0, 1), inputs[1])



if __name__ == "__main__":
    app = QApplication(sys.argv)
    m_net = Net()
    m_training_data = TrainingData()
    m_controller = Controller()
    print("EOF ")
    sys.exit(app.exec_())





