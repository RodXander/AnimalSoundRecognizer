__author__ = 'RodXander'

from PyQt4.QtGui import *
from PyQt4.QtCore import *


class OptionDlg(QDialog):

    def __init__(self, state, parent=None):
        super(OptionDlg, self).__init__(parent)

        self.save_line_edit = QLineEdit()
        self.state = state
        self.save_line_edit.setText(state.last_save_dir)

        self.examine_button = QPushButton('Examinar...')
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.save_line_edit)
        save_layout.addWidget(self.examine_button)

        middle_layout = QGridLayout()

        middle_layout.addWidget(QLabel('Salvar en:'), 0, 0)
        middle_layout.addLayout(save_layout, 1, 0, 1, 2)

        self.state_number = QSpinBox()
        self.state_number.setMinimum(1)
        middle_layout.addWidget(self.state_number, 2, 0)
        middle_layout.addWidget(QLabel(unicode('N\xa3mero de estados', encoding='CP850')), 2, 1)

        self.component_number = QSpinBox()
        self.component_number.setMinimum(1)
        middle_layout.addWidget(self.component_number, 3, 0)
        middle_layout.addWidget(QLabel(unicode('N\xa3mero de componentes por estados', encoding='CP850')), 3, 1)

        self.max_segmentations = QSpinBox()
        self.max_segmentations.setMinimum(1)
        self.max_segmentations.setMaximum(500)
        middle_layout.addWidget(self.max_segmentations, 4, 0)
        middle_layout.addWidget(QLabel(unicode('N\xa3mero m\xa0ximo de segmentaciones', encoding='CP850')), 4, 1)


        self.perform_baum_welch = QCheckBox('Realizar procedimiento Baum-Welch')
        middle_layout.addWidget(self.perform_baum_welch, 5, 1)

        self.nbaum_welch_iterations = QSpinBox()
        self.nbaum_welch_iterations.setMinimum(1)
        middle_layout.addWidget(self.nbaum_welch_iterations, 6, 0)
        middle_layout.addWidget(QLabel(unicode('N\xa3mero de iteraciones del Baum-Welch', encoding='CP850')), 6, 1)

        middle_layout.addWidget(QLabel(), 7, 0, 1, 2)


        self.default_values = QPushButton('Valores predeterminados')
        self.accept_button = QPushButton('Aceptar')
        self.cancel_button = QPushButton('Cancelar')
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.default_values)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.accept_button)
        buttons_layout.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addLayout(middle_layout)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

        self.setWindowTitle('Opciones del modelo')
        self.setFixedSize(350, 240)
        self.set_values(defaults=False)

        self.connect(self.examine_button, SIGNAL('clicked()'), self.search_dir_dialog)
        self.connect(self.default_values, SIGNAL('clicked()'), self.set_values)
        self.connect(self.cancel_button, SIGNAL('clicked()'), self.close)
        self.connect(self.accept_button, SIGNAL('clicked()'), self.accept_wrapper)
        self.connect(self.perform_baum_welch, SIGNAL('clicked()'), self.not_perform_bw)

    def search_dir_dialog(self):
        directory = QFileDialog().getSaveFileName(self, 'Salvar el nuevo modelo...', self.state.last_save_dir, 'Archivos de reconocimiento (*.rdx)')

        if directory != '':
            self.state.last_save_dir = str(directory)
            self.save_line_edit.setText(directory)

    def set_values(self, defaults=True):
        if defaults:
            self.state.default_model_values()

        self.nbaum_welch_iterations.setValue(self.state.niterations_bw)
        self.perform_baum_welch.setChecked(self.state.make_bw)
        self.max_segmentations.setValue(self.state.nsegmentations)
        self.component_number.setValue(self.state.ncomponents)
        self.state_number.setValue(self.state.nstates)
        self.nbaum_welch_iterations.setDisabled(not self.perform_baum_welch.isChecked())

    def not_perform_bw(self):
        if self.sender().isChecked():
            self.nbaum_welch_iterations.setDisabled(False)
        else:
            self.nbaum_welch_iterations.setDisabled(True)

    def accept_wrapper(self):
        self.state.nstates = self.state_number.value()
        self.state.ncomponents = self.component_number.value()
        self.state.nsegmentations = self.max_segmentations.value()
        self.state.make_bw = self.perform_baum_welch.isChecked()
        self.state.niterations_bw = self.nbaum_welch_iterations.value()

        self.accept()


class AboutDlg(QDialog):

    def __init__(self, parent=None):
        super(AboutDlg, self).__init__(parent)

        self.image = QImage('Degree.png')
        self.image_label = QLabel()
        self.image_label.setFixedSize(48, 48)
        self.image_label.setPixmap(QPixmap.fromImage(self.image))
        self.name_label = QLabel('<span style="font-weight: bold; font-size: 18px;">Reconocedor de audio<\span>')
        self.name_label.setContentsMargins(10, 0, 0, 0)

        layout_img_name = QHBoxLayout()
        layout_img_name.addWidget(self.image_label)
        layout_img_name.addWidget(self.name_label)

        self.text = QLabel(unicode('Aplicaci\xa2n demonstrativa en opci\xa2n \n      al grado de Licenciado en \n       Ciencia de la Computaci\xa2n', encoding='cp850'))
        self.text.setContentsMargins(0, 3, 0, 0)

        layout_text = QHBoxLayout()
        layout_text.addStretch()
        layout_text.addWidget(self.text)
        layout_text.addStretch()

        my_info_label = QLabel(unicode('<b>Dise\xa4o y programaci\xa2n:</b> Osvaldo A. Saez Lombira', encoding='cp850'))
        my_info_label.setContentsMargins(0, 12, 0, 0)
        tutor_info_label = QLabel('<b>Tutores:</b> Alfredo Somoza y Emanuel Mora')

        layout_my_info = QHBoxLayout()
        layout_my_info.addStretch()
        layout_my_info.addWidget(my_info_label)
        layout_my_info.addStretch()

        layout_tutor_info = QHBoxLayout()
        layout_tutor_info.addStretch()
        layout_tutor_info.addWidget(tutor_info_label)
        layout_tutor_info.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(layout_img_name)
        layout.addLayout(layout_text)
        layout.addLayout(layout_my_info)
        layout.addLayout(layout_tutor_info)

        self.setLayout(layout)
        self.setWindowTitle('Acerca')
        self.setFixedSize(300, 175)


class WarningDlg(QDialog):

    def __init__(self, text, parent=None):
        super(WarningDlg, self).__init__(parent)

        layout_text = QHBoxLayout()
        layout_text.addStretch()
        layout_text.addWidget(QLabel(text))
        layout_text.addStretch()

        accept_button = QPushButton('Aceptar')
        layout_button = QHBoxLayout()
        layout_button.addStretch()
        layout_button.addWidget(accept_button)

        layout = QVBoxLayout()
        layout.addLayout(layout_text)
        layout.addLayout(layout_button)
        self.setLayout(layout)

        self.connect(accept_button, SIGNAL('clicked()'), self.close)
        self.setFixedSize(300, 100)
        self.setWindowTitle('Aviso')