from PyQt5.QtWidgets import *


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.button = QPushButton('Test', self)
        self.label = QLabel(self)
        self.button.clicked.connect(self.handleButton)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.button)

    def handleButton(self):
        self.label.setText('Button Clicked!')


class Node(object):
    def __init__(self, d, n):
        self.data = d
        self.next_node = n


if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
