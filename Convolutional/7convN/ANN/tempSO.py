from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Controller(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(700, 100, 905, 700)