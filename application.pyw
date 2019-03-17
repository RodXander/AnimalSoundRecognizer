__author__ = 'RodXander'

import sys
from ui.main_window import MainWindow
from PyQt4.QtCore import *
from PyQt4.QtGui import *

app = QApplication(sys.argv)

app.setWindowIcon(QIcon('Degree_16.ico'))

main_form = MainWindow()
main_form.show()

app.exec_()
