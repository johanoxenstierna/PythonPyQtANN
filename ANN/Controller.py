
import sys, os

from functools import partial

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from ANN.Net import Net


class Controller(QMainWindow):
    def __init__(self):
        super().__init__()


        self.setGeometry(700, 100, 905, 700)

        pic = QLabel(self)
        pic.setGeometry(0, 0, 900, 700)
        pic.setPixmap(QPixmap(os.getcwd() + "/myPic.png"))


        self.l0inputs = []
        self.l0outputs = []
        self.l0wsum = QLineEdit(self)

        self.synapse_buttonsL0L1 = {}

        self.l1inputs = []
        self.l1outputs = []
        self.l1wsum = QLineEdit(self)

        self.synapse_buttonsL1L2 = {}

        self.l2inputs = []
        self.l2outputs = []

        self.expected_v = QLineEdit(self)
        self.delta_v = QLineEdit(self)
        self.delta_gradient_v = QLineEdit(self)
        self.MSE_v = QLineEdit(self)
        self.epochs_v = QLineEdit(self)
        self.mse_all_data_v = QLineEdit(self)
        self.end_of_data_press_again_v = QLabel(self)

        self.delta_printbox = QTextEdit(self)

        self.button_group = QButtonGroup()

        self.load_inputs_button = QPushButton(self)
        self.load_inputs_button.clicked.connect(m_net.load_inputs)
        self.load_inputs_button.setCheckable(True)
        self.button_group.addButton(self.load_inputs_button)

        self.forward_propL0L1_button = QPushButton(self)
        self.forward_propL0L1_button.clicked.connect(m_net.forward_propL0L1)
        self.forward_propL0L1_button.setCheckable(True)
        self.button_group.addButton(self.forward_propL0L1_button)

        self.forward_propL1L2_button = QPushButton(self)
        self.forward_propL1L2_button.clicked.connect(m_net.forward_propL1L2)
        self.forward_propL1L2_button.setCheckable(True)
        self.button_group.addButton(self.forward_propL1L2_button)

        self.calculate_MSE_and_deltagrad_button = QPushButton(self)
        self.calculate_MSE_and_deltagrad_button.clicked.connect(m_net.calculate_MSE_and_deltagradient)
        self.calculate_MSE_and_deltagrad_button.setCheckable(True)
        self.button_group.addButton(self.calculate_MSE_and_deltagrad_button)

        self.back_propL2L1_button = QPushButton(self)
        self.back_propL2L1_button.clicked.connect(m_net.back_propL2L1)
        self.back_propL2L1_button.setCheckable(True)
        self.button_group.addButton(self.back_propL2L1_button)

        self.back_propL1L0_button = QPushButton(self)
        self.back_propL1L0_button.clicked.connect(m_net.back_propL1L0)
        self.back_propL1L0_button.setCheckable(True)
        self.button_group.addButton(self.back_propL1L0_button)

        self.updateP_weights_button = QPushButton(self)
        self.updateP_weights_button.clicked.connect(partial(m_net.update_weights, "deltaweight"))
        self.updateP_weights_button.setCheckable(True)
        self.button_group.addButton(self.updateP_weights_button)

        self.run_epoch_button = QPushButton(self)
        self.run_epoch_button.clicked.connect(m_net.run_epoch)
        self.run_epoch_button.setCheckable(True)
        self.button_group.addButton(self.run_epoch_button)

        self.run_100_epochs_pattern_button = QPushButton(self)
        self.run_100_epochs_pattern_button.clicked.connect(partial(self.run_100_epochs, "deltaweight"))
        self.run_100_epochs_pattern_button.setCheckable(True)
        self.button_group.addButton(self.run_100_epochs_pattern_button)

        self.run_100_epochs_batch_button = QPushButton(self)
        self.run_100_epochs_batch_button.clicked.connect(partial(self.run_100_epochs, "batch_deltaweight"))
        self.run_100_epochs_batch_button.setCheckable(True)
        self.button_group.addButton(self.run_100_epochs_batch_button)


        self.initialize_and_draw_lineEdits_and_buttons()
        self.initialize_buttons()

        for i in range(0, 3):
            for j in range(0, 4):
                self.synapse_buttonsL0L1[i, j].clicked.connect(partial(self.delta_print, 0, i, j))

        for k in range(0, 5):
            self.synapse_buttonsL1L2[k, 0].clicked.connect(partial(self.delta_print, 1, k, 0))


        self.show()


    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_neurons(qp)
        self.draw_synapses_and_labels(qp)
        self.update_lineEdits_and_buttons()

        qp.end()

    def draw_neurons(self, qp):
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        #draw first layer
        qp.setBrush(QColor(242, 231, 138))
        for i in range(0, 3):
            c = m_net.get_coordinates(m_net.get_neuron(0, i))
            qp.drawEllipse(c[0] - 25, c[1] - 25, 50, 50)

        # draw second layer
        qp.setBrush(QColor(147, 218, 249))
        for i in range(0, 5):
            c = m_net.get_coordinates(m_net.get_neuron(1, i))
            qp.drawEllipse(c[0] - 25, c[1] - 25, 50, 50)

        # draw third layer
        qp.setBrush(QColor(249, 203, 203))
        for i in range(0, 2):
            c = m_net.get_coordinates(m_net.get_neuron(2, i))
            qp.drawEllipse(c[0] - 25, c[1] - 25, 50, 50)

    def draw_synapses_and_labels(self, qp):
        pen = QPen(Qt.black, 1, Qt.DashLine)
        qp.setPen(pen)

        # L0L1
        qp.drawLine(175, 225, 205, 200)
        qp.drawLine(175, 225, 205, 220)
        qp.drawLine(175, 225, 205, 240)
        qp.drawLine(175, 225, 205, 260)

        qp.drawLine(175, 325, 205, 295)
        qp.drawLine(175, 325, 205, 315)
        qp.drawLine(175, 325, 205, 335)
        qp.drawLine(175, 325, 205, 355)

        qp.drawLine(175, 425, 205, 395)
        qp.drawLine(175, 425, 205, 415)
        qp.drawLine(175, 425, 205, 435)
        qp.drawLine(175, 425, 205, 455)

        qp.drawLine(260, 195, 310, 125)
        qp.drawLine(260, 220, 310, 225)
        qp.drawLine(260, 240, 310, 325)
        qp.drawLine(260, 260, 310, 425)

        qp.drawLine(260, 295, 310, 125)
        qp.drawLine(260, 320, 310, 225)
        qp.drawLine(260, 340, 310, 325)
        qp.drawLine(260, 360, 310, 425)

        qp.drawLine(260, 395, 310, 125)
        qp.drawLine(260, 420, 310, 225)
        qp.drawLine(260, 440, 310, 325)
        qp.drawLine(260, 460, 310, 425)

        # L1L2
        qp.drawLine(460, 125, 510, 125)
        qp.drawLine(460, 225, 510, 225)
        qp.drawLine(460, 325, 510, 325)
        qp.drawLine(460, 425, 510, 425)
        qp.drawLine(460, 525, 510, 525)

        qp.drawLine(555, 125, 610, 275)
        qp.drawLine(555, 225, 610, 275)
        qp.drawLine(555, 325, 610, 275)
        qp.drawLine(555, 425, 610, 275)
        qp.drawLine(555, 525, 610, 275)

        qp.drawText(10, 210, "input")
        qp.drawText(125, 210, "output")
        qp.drawText(210, 180, "synapses")

        qp.drawText(310, 110, "input")
        qp.drawText(425, 110, "output")
        qp.drawText(510, 110, "synapses")

        qp.drawText(610, 260, "input")
        qp.drawText(730, 260, "output")
        qp.drawText(780, 260, "expected")
        qp.drawText(850, 260, "delta")
        qp.drawText(780, 305, "this pat MSE")
        qp.drawText(850, 305, "deltagrad")
        qp.drawText(780, 350, "batch MSE")

        pen2 = QPen(Qt.black, 1, Qt.SolidLine)
        qp.setPen(pen2)
        qp.setFont(QFont("Times", 14))
        qp.drawText(87, 190, "L0")
        qp.drawText(390, 90, "L1")
        qp.drawText(690, 240, "L2")

        qp.drawText(92, 235, "0")
        qp.drawText(92, 335, "1")
        qp.drawText(92, 435, "B")
        qp.drawText(395, 135, "0")
        qp.drawText(395, 235, "1")
        qp.drawText(395, 335, "2")
        qp.drawText(395, 435, "3")
        qp.drawText(395, 535, "B")
        qp.drawText(695, 285, "0")

        self.delta_printbox.setFont(QFont("Times", 12))

    def initialize_buttons(self):
        self.load_inputs_button.setGeometry(10, 570, 100, 60)
        self.load_inputs_button.setFont(QFont("Times", 12))
        self.load_inputs_button.setText("Load inp")

        self.forward_propL0L1_button.setGeometry(200, 570, 100, 60)
        self.forward_propL0L1_button.setFont(QFont("Times", 12))
        self.forward_propL0L1_button.setText("F-propL0L1")

        self.forward_propL1L2_button.setGeometry(500, 570, 100, 60)
        self.forward_propL1L2_button.setFont(QFont("Times", 12))
        self.forward_propL1L2_button.setText("F-propL1L2")

        self.calculate_MSE_and_deltagrad_button.setGeometry(700, 570, 100, 60)
        self.calculate_MSE_and_deltagrad_button.setFont(QFont("Times", 12))
        self.calculate_MSE_and_deltagrad_button.setText("Calc errors")

        self.back_propL2L1_button.setGeometry(500, 630, 100, 60)
        self.back_propL2L1_button.setFont(QFont("Times", 12))
        self.back_propL2L1_button.setText("B-propL2L1")

        self.back_propL1L0_button.setGeometry(200, 630, 100, 60)
        self.back_propL1L0_button.setFont(QFont("Times", 12))
        self.back_propL1L0_button.setText("B-propL1L0")

        self.updateP_weights_button.setGeometry(10, 630, 100, 60)
        self.updateP_weights_button.setFont(QFont("Times", 12))
        self.updateP_weights_button.setText("Update P")

        self.run_epoch_button.setGeometry(0, 0, 100, 40)
        self.run_epoch_button.setFont(QFont("Times", 12))
        self.run_epoch_button.setText("Run epoch")

        self.run_100_epochs_batch_button.setGeometry(102, 0, 210, 40)
        self.run_100_epochs_batch_button.setFont(QFont("Times", 12))
        self.run_100_epochs_batch_button.setText("Run 100 epochs with batch")

        self.run_100_epochs_pattern_button.setGeometry(102, 42, 210, 40)
        self.run_100_epochs_pattern_button.setFont(QFont("Times", 12))
        self.run_100_epochs_pattern_button.setText("Run 100 epochs with pattern")

    def initialize_and_draw_lineEdits_and_buttons(self):

        # first layer and L0L1
        for i in range(0, 2):
            c = m_net.get_coordinates(m_net.get_neuron(0, i))

            # inputs
            my_line_edit_in = QLineEdit(self)
            my_line_edit_in.setGeometry(c[0] - 90, c[1] - 10, 60, 20)
            self.l0inputs.append([i, my_line_edit_in])

            # outputs
            my_line_edit_out = QLineEdit(self)
            my_line_edit_out.setGeometry(c[0] + 25, c[1] - 10, 50, 20)
            self.l0outputs.append([i, my_line_edit_out])

            # weights
            for j in range(0, 4):

                my_line_edit_w = QPushButton(self)
                my_line_edit_w.setGeometry(c[0] + 100, c[1] - 40 + (22 * j), 60, 20)
                self.synapse_buttonsL0L1[i, j] = my_line_edit_w

        #bias L0L1
        cb = m_net.get_coordinates(m_net.get_neuron(0, 2))

        #bias output
        my_line_edit_bout = QLineEdit(self)
        my_line_edit_bout.setGeometry(cb[0] + 25, cb[1] - 10, 50, 20)
        self.l0outputs.append([2, my_line_edit_bout])
        #bias weights
        for i in range(0, 4):
            my_line_edit_bw = QPushButton(self)
            my_line_edit_bw.setGeometry(cb[0] + 100, cb[1] - 40 + (22 * i), 60, 20)
            self.synapse_buttonsL0L1[2, i] = my_line_edit_bw


        # second layer and L1L2
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
            my_line_edit_w = QPushButton(self)
            my_line_edit_w.setGeometry(c[0] + 100, c[1] - 10, 60, 20)
            self.synapse_buttonsL1L2[i, 0] = my_line_edit_w

        #bias L1L2
        cb = m_net.get_coordinates(m_net.get_neuron(1, 4))
        my_line_edit_bout = QLineEdit(self)
        my_line_edit_bout.setGeometry(cb[0] + 25, cb[1] - 10, 50, 20)
        self.l1outputs.append([5, my_line_edit_bout])
        my_line_edit_bw = QPushButton(self)
        my_line_edit_bw.setGeometry(cb[0] + 100, cb[1] - 10, 60, 20)
        self.synapse_buttonsL1L2[4, 0] = my_line_edit_bw

        # third layer
        for i in range(0, 1):
            c = m_net.get_coordinates(m_net.get_neuron(2, i))

            # inputs
            my_line_edit_in = QLineEdit(self)
            my_line_edit_in.setGeometry(c[0] - 90, c[1] - 10, 60, 20)
            self.l2inputs.append([i, my_line_edit_in])

            # outputs
            my_line_edit_out = QLineEdit(self)
            my_line_edit_out.setGeometry(c[0] + 25, c[1] - 10, 50, 20)
            self.l2outputs.append([i, my_line_edit_out])

        # draw expected output and error
        self.expected_v.setGeometry(780, 265, 50, 20)
        self.delta_v.setGeometry(840, 265, 50, 20)
        self.MSE_v.setGeometry(780, 310, 50, 20)
        self.delta_gradient_v.setGeometry(840, 310, 50, 20)
        self.delta_printbox.setGeometry(650, 10, 220, 50)
        self.epochs_v.setGeometry(460, 10, 60, 30)
        self.epochs_v.setFont(QFont("Times", 15))
        self.mse_all_data_v.setGeometry(780, 355, 50, 20)
        self.end_of_data_press_again_v.setGeometry(5, 520, 200, 30)

    def update_lineEdits_and_buttons(self):

        #update first layer
        synapses = getattr(m_net, 'synapsesL0L1')

        for i in range(0, 2):
            #input
            input = m_net.get_input(m_net.get_neuron(0, i))
            QLineEdit.setText(self.l0inputs[i][1], str(round(input, 3)))

            #outputs
            output = m_net.get_output(m_net.get_neuron(0, i))
            QLineEdit.setText(self.l0outputs[i][1], str(round(output, 3)))

            #weights
            for j in range(0, 4):
                weight = getattr(synapses[i, j], 'weight')
                QPushButton.setText(self.synapse_buttonsL0L1[i, j], str(round(weight, 4)))

        #bias
        bias_o = m_net.get_output(m_net.get_neuron(0, 2))
        QLineEdit.setText(self.l0outputs[2][1], str(bias_o))
        for i in range(0, 4):
            bias_w = getattr(synapses[2, i], 'weight')
            QPushButton.setText(self.synapse_buttonsL0L1[2, i], str(round(bias_w, 4)))


        # update second layer
        synapses = getattr(m_net, 'synapsesL1L2')

        for i in range(0, 4):

            # input
            input = m_net.get_input(m_net.get_neuron(1, i))
            QLineEdit.setText(self.l1inputs[i][1], str(round(input, 3)))

            # outputs
            output = m_net.get_output(m_net.get_neuron(1, i))
            QLineEdit.setText(self.l1outputs[i][1], str(round(output, 3)))

            # weights
            weight = getattr(synapses[i, 0], 'weight')
            QPushButton.setText(self.synapse_buttonsL1L2[i, 0], str(round(weight, 4)))

        #bias
        bias_o = m_net.get_output(m_net.get_neuron(1, 4))
        QLineEdit.setText(self.l1outputs[4][1], str(bias_o))
        bias_w = getattr(synapses[4, 0], 'weight')
        QPushButton.setText(self.synapse_buttonsL1L2[4, 0], str(round(bias_w, 4)))

        # update third layer

        for i in range(0, 1):

            # input
            input = m_net.get_input(m_net.get_neuron(2, i))
            QLineEdit.setText(self.l2inputs[i][1], str(round(input, 3)))

            # outputs
            output = m_net.get_output(m_net.get_neuron(2, i))
            QLineEdit.setText(self.l2outputs[i][1], str(round(output, 3)))

        QLineEdit.setText(self.expected_v, str(round(getattr(m_net, 'expected'), 3)))
        QLineEdit.setText(self.MSE_v, str(round(getattr(m_net, 'MSE'), 3)))
        QLineEdit.setText(self.delta_v, str(round(getattr(m_net, 'delta'), 3)))
        QLineEdit.setText(self.delta_gradient_v, str(round(getattr(m_net, 'deltagradientL2'), 3)))

        QLineEdit.setText(self.epochs_v, str(round(getattr(m_net, 'epoch'), 3)))
        QLineEdit.setText(self.mse_all_data_v, str(round(getattr(m_net, 'batch_MSE'), 3)))
        self.end_of_data_press_again_v.setText("EOF, press twice to load from top!"
                                               if m_net.end_of_data_press_again == True else "")

    def delta_print(self, layerIndex, i, j):
        if layerIndex == 0:
            deltaweight = getattr(m_net.synapsesL0L1[i, j], 'deltaweight')
            batch_deltaweight = getattr(m_net.synapsesL0L1[i, j], 'batch_deltaweight')
            self.delta_printbox.setText("Delta_w: " + str(round(deltaweight, 8)))
            self.delta_printbox.append("Batch delta_w: " + str(round(batch_deltaweight, 8)))


        else:
            deltaweight = getattr(m_net.synapsesL1L2[i, j], 'deltaweight')
            batch_deltaweight = getattr(m_net.synapsesL1L2[i, j], 'batch_deltaweight')
            self.delta_printbox.setText("Delta_w: " + str(round(deltaweight, 8)))
            self.delta_printbox.append("Batch delta_w: " + str(round(batch_deltaweight, 8)))

    def run_100_epochs(self, update_type_controller):
        if update_type_controller == "deltaweight":
            m_net.update_type = "deltaweight"
        else:
            m_net.update_type = "batch_deltaweight"
        for i in range(0, 100):
            m_net.run_epoch()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    m_net = Net()
    m_controller = Controller()
    print("EOF ")
    sys.exit(app.exec_())






