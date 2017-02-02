

import sys, random, math

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
# from TheANNXor.Neuron import Neuron
from TheANNXor.Net import Net, Synapse
from TheANNXor.Layer import Layer
from six import with_metaclass
from TheANNXor.TrainingData import TrainingData



class Controller(QMainWindow):
    # here is where everything will be set
    def __init__(self):
        super().__init__()

        #initialize window
        self.text = "ggg"
        self.resize(700, 700)
        self.setGeometry(700, 100, 900, 700)

        #instance vars in window (this is what will be updated dynamically)

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
        #
        # #create and set up events for buttons
        self.load_inputs_button = QPushButton(self)
        self.load_inputs_button.clicked.connect(self.init_and_draw_next_inputs)
        self.forward_propL0_button = QPushButton(self)
        self.forward_propL0_button.clicked.connect(self.forward_propL0)
        self.forward_propL1_button = QPushButton(self)
        self.forward_propL1_button.clicked.connect(self.forward_propL1)
        self.forward_propL2_button = QPushButton(self)
        self.forward_propL2_button.clicked.connect(self.forward_propL2)
        self.back_prop_button = QPushButton(self)
        self.back_prop_button.clicked.connect(self.back_prop)
        self.initialize_buttons()




        self.show()

    # PAINTEVENT ##########################################
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        #self.drawText(event, qp)
        self.drawPoints(qp)
        self.draw_neurons(qp)
        self.draw_synapses_and_labels(qp)
        self.update_lineEdits()

        qp.end()

    def draw_neurons(self, qp):

        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        # draw first layer
        for i in range(0, 3):
            c = m_net.get_coordinates(m_net.get_neuron(0, i))
            qp.drawEllipse(c[0] - 25, c[1] - 25, 50, 50)

        #draw second layer
        for i in range(0, 5):
            c = m_net.get_coordinates(m_net.get_neuron(1, i))
            qp.drawEllipse(c[0] - 25, c[1] - 25, 50, 50)

        #draw third layer
        for i in range(0, 2):
            c = m_net.get_coordinates(m_net.get_neuron(2, i))
            qp.drawEllipse(c[0] - 25, c[1] - 25, 50, 50)

    def initialize_buttons(self):
        self.load_inputs_button.setGeometry(10, 500, 100, 60)
        self.load_inputs_button.setText("Load inp")
        self.forward_propL0_button.setGeometry(10, 570, 100, 60)
        self.forward_propL0_button.setText("F-propL0")
        self.forward_propL1_button.setGeometry(300, 570, 100, 60)
        self.forward_propL1_button.setText("F-propL1")
        self.forward_propL2_button.setGeometry(600, 570, 100, 60)
        self.forward_propL2_button.setText("F-propL2")
        self.back_prop_button.setGeometry(390, 650, 160, 60)
        self.back_prop_button.setText("B-prop")

    def initialize_and_draw_lineEdits(self):

        #this method DOES NOT collect data inside neurons so dont put in testing data here

        # set & draw first layer Do this last in 3
        for i in range(0, 2):

            c = m_net.get_coordinates(m_net.get_neuron(0, i))

            # inputs
            my_line_edit_in = QLineEdit(self)
            my_line_edit_in.setGeometry(c[0] - 90, c[1] - 10, 60, 20)
            self.l0inputs.append([i, my_line_edit_in])

            # outputs
            my_line_edit_out = QLineEdit(self)
            my_line_edit_out.setGeometry(c[0] + 25, c[1] - 40, 50, 85)
            self.l0outputs.append(my_line_edit_out)

            for j in range(0, 4):
                #weights
                my_line_edit_w = QLineEdit(self)
                my_line_edit_w.setGeometry(c[0] + 100, c[1] - 40 + (22 * j), 50, 20)
                self.l0l1weights[i, j] = my_line_edit_w

        # create lineedit for w_sum.
        c = m_net.get_coordinates(m_net.get_neuron(0, 0))
        self.l0wsum.setGeometry(c[0] + 100, c[1] - 80, 50, 20)


        # set & draw second layer
        for i in range(0, 4):
            c = m_net.get_coordinates(m_net.get_neuron(1, i))

            # inputs
            my_line_edit_in = QLineEdit(self)
            my_line_edit_in.setGeometry(c[0] - 90, c[1] - 10, 60, 20)
            self.l1inputs.append([i, my_line_edit_in])

            # outputs
            my_line_edit_out = QLineEdit(self)
            my_line_edit_out.setGeometry(c[0] + 25, c[1] - 10, 50, 20)
            self.l1outputs.append([i, my_line_edit_out])

            # weights
            my_line_edit_w = QLineEdit(self)
            my_line_edit_w.setGeometry(c[0] + 100, c[1] - 10, 50, 20)
            self.l1l2weights[i] = my_line_edit_w

        # create lineedit for w_sum.
        c = m_net.get_coordinates(m_net.get_neuron(1, 0))
        self.l1wsum = QLineEdit(self)
        self.l1wsum.setGeometry(c[0] + 100, c[1] - 50, 50, 20)

        #set & draw third layer
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

        # self.update_lineEdits()

    def update_lineEdits(self):

        #this function is called in message loop.
        # and updates everything in the window that starts with Q

        #update first layer

        weights_sum0 = 0
        synapses = getattr(m_net, 'synapsesl0l1')

        for i in range(0, 2):
            #inputs
            input = m_net.get_input(m_net.get_neuron(0, i))
            QLineEdit.setText(self.l0inputs[i][1], str(input))

            # outputs
            output = m_net.get_output(m_net.get_neuron(0, i))
            QLineEdit.setText(self.l0outputs[i], str(round(output, 3)))

            # weights
            for j in range(0, 4):
                weight = getattr(synapses[i, j], 'weight')
                QLineEdit.setText(self.l0l1weights[i, j], str(weight))
                weights_sum0 += weight


        QLineEdit.setText(self.l0wsum, str(weights_sum0))



        #update second layer
        weights_sum1 = 0
        synapses = getattr(m_net, 'synapsesl1l2')

        #for each neuron
        for i in range(0, 4):
            # inputs
            input = m_net.get_input(m_net.get_neuron(1, i))
            QLineEdit.setText(self.l1inputs[i][1], str(round(input, 4)))

            # outputs
            output = m_net.get_output(m_net.get_neuron(1, i))
            QLineEdit.setText(self.l1outputs[i][1], str(round(output, 3)))

            weight = getattr(synapses[i, 0], 'weight')
            QLineEdit.setText(self.l1l2weights[i], str(weight))

        QLineEdit.setText(self.l1wsum, str(weights_sum1))


        # update third layer
        for i in range(0, 1):
            # inputs
            input = m_net.get_input(m_net.get_neuron(2, i))
            QLineEdit.setText(self.l2inputs[i][1], str(input))

            # outputs
            output = m_net.get_output(m_net.get_neuron(2, i))
            QLineEdit.setText(self.l2outputs[i][1], str(round(output, 3)))
        #update net variables
        QLineEdit.setText(self.expected_v, str(m_net.expected))
        QLineEdit.setText(self.error_v, str(m_net.error))

    def draw_synapses_and_labels(self, qp):
        pen = QPen(Qt.black, 1, Qt.DashLine)
        qp.setPen(pen)

        qp.drawLine(170, 200, 200, 200)
        qp.drawLine(170, 220, 200, 220)
        qp.drawLine(170, 240, 200, 240)
        qp.drawLine(170, 260, 200, 260)

        qp.drawLine(170, 295, 200, 295)
        qp.drawLine(170, 315, 200, 315)
        qp.drawLine(170, 335, 200, 335)
        qp.drawLine(170, 355, 200, 355)

        qp.drawLine(250, 200, 325, 125)
        qp.drawLine(250, 220, 325, 225)
        qp.drawLine(250, 240, 325, 325)
        qp.drawLine(250, 260, 325, 425)

        qp.drawLine(250, 300, 325, 125)
        qp.drawLine(250, 320, 325, 225)
        qp.drawLine(250, 340, 325, 325)
        qp.drawLine(250, 360, 325, 425)

        qp.drawLine(520, 125, 625, 275)
        qp.drawLine(520, 225, 625, 275)
        qp.drawLine(520, 325, 625, 275)
        qp.drawLine(520, 425, 625, 275)

        qp.drawLine(460, 125, 500, 125)
        qp.drawLine(460, 225, 500, 225)
        qp.drawLine(460, 325, 500, 325)
        qp.drawLine(460, 425, 500, 425)

        qp.drawText(125, 180, "out")
        qp.drawText(200, 180, "w's")
        qp.drawText(200, 140, "sum w")

        qp.drawText(425, 110, "out")
        qp.drawText(500, 110, "w's")
        qp.drawText(500, 70, "sum w")

        qp.drawText(725, 260, "out")
        qp.drawText(780, 260, "expc")
        qp.drawText(845, 260, "err")

    def init_and_draw_next_inputs(self):
        #this is highly hard-coded
        inputs = m_training_data.get_next_inputs()

        #model
        m_net.set_input(m_net.get_neuron(0,0), inputs[0])
        m_net.set_input(m_net.get_neuron(0,1), inputs[1])

        #viewer
        QLineEdit.setText(self.l0inputs[0][1], inputs[0])

    def forward_propL0(self):
        #all were gonna do here for now is to put first layers inputs into its outputs

        #get prev layer
        layer = m_net.get_layer(0)

        #for all the neurons in previous layer
        for i in range(0, 2):
            neuron = layer.get_neuron_from_layer(i)
            nn = m_net.get_neuron(0, 0)

            new_output = m_net.get_input(neuron)
            m_net.set_output(neuron, new_output)

    def forward_propL1(self):

        #for each neuron in layer 1
        for i in range(0, 4):

            sum = 0

            #for each neuron in layer 0
            for j in range(0, 2):
                prev_neuron = m_net.get_neuron(0, j)
                prev_neuron_output = m_net.get_output(prev_neuron)
                weight = m_net.get_weight(0, j, i)

                sum += prev_neuron_output * weight

            neuron1 = m_net.get_neuron(1, i)
            m_net.set_input(neuron1, str(sum))

            n_output = self.transfer_function(sum)
            m_net.set_output(neuron1, n_output)



        #save above as example. first started doing rounding here but then realized rounding should be donw in viewer exclusively.

    def forward_propL2(self):

        sum = 0

        neuron2 = m_net.get_neuron(2, 0)

        # #for each neuron in layer 0
        # for j in range(0, 2):
        #     prev_neuron = m_net.get_neuron(0, j)
        #     prev_neuron_output = m_net.get_output(prev_neuron)
        #     weight = m_net.get_weight(0, j, i)
        #
        #     sum += prev_neuron_output * weight
        #
        # neuron1 = m_net.get_neuron(2, 0)
        # m_net.set_input(neuron1, str(sum))
        #
        # n_output = self.transfer_function(sum)
        # m_net.set_output(neuron1, n_output)



        #save above as example. first started doing rounding here but then realized rounding should be donw in viewer exclusively.

    def transfer_function(self, x):
        #use hyperbolic tan
        return math.tanh(x)

    def back_prop(self):
        pass

    def drawText(self, event, qp):
        qp.setPen(QColor(168, 34, 3))
        qp.setFont(QFont('Decorative', 10))
        qp.drawText(event.rect(), Qt.AlignCenter, self.text)

    def drawPoints(self, qp):
        qp.setPen(Qt.red)
        size = self.size()

        for i in range(1000):
            x = random.randint(1, size.width() - 1)
            y = random.randint(1, size.height() - 1)
            qp.drawPoint(x, y)




# this is the main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    m_training_data = TrainingData()
    m_net = Net()
    ex = Controller()
    print("EOF controller")
    sys.exit(app.exec_())






