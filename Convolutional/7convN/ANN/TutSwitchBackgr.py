import sys, os
from PyQt5 import QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Example(QMainWindow):

    def __init__(self):
        super(Example, self).__init__()
        self.initUI()

    def initUI(self):
        # pic = QLabel(self)
        # pic.setGeometry(0, 0, 900, 700)
        # pic.setPixmap(QPixmap(os.getcwd() + "/myPicT2.png"))

        self.setToolTip('This is a <b>QWidget</b> widget')

        # Show  image
        self.pic = QLabel(self)
        self.pic.setGeometry(10, 10, 800, 800)
        self.pic.setPixmap(QPixmap(os.getcwd() + "/myPicT2.png"))

        # Show button
        btn = QPushButton('Button', self)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        btn.resize(btn.sizeHint())
        btn.clicked.connect(self.fun)
        btn.move(50, 50)


        self.setGeometry(300, 300, 2000, 1500)
        self.setWindowTitle('Tooltips')
        self.show()

    # Connect button to image updating
    def fun(self):
        self.pic.setPixmap(QPixmap(os.getcwd() + "/myPic.png"))

def main():

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()