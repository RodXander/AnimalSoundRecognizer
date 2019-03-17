__author__ = 'RodXander'

import os
import os.path

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import binding
from dialogs import OptionDlg, AboutDlg, WarningDlg




# Binds the UI information with the logic and viceversa. Also for saving some settings.
state = binding.AppState()


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setWindowTitle('Reconocedor de audio')
        self.main_widget = MainWidget()
        self.menu_bar = self.menuBar()

        # Actions
        add_audio_action = QAction(unicode('A\xa4adir nuevo audio...', encoding='cp850'), self)
        add_model_action = QAction(unicode('A\xa4adir nuevo modelo...', encoding='cp850'), self)
        erase_all_audios_action = QAction(unicode('Borrar todos los audios', encoding='cp850'), self)
        erase_all_models_action = QAction(unicode('Borrar todos los modelos', encoding='cp850'), self)

        close_action = QAction('Cerrar', self)
        about_action = QAction('About', self)

        # Connections of the actions
        self.connect(add_audio_action, SIGNAL('triggered()'), self.main_widget.audio_widget.add_audio)
        self.connect(add_model_action, SIGNAL('triggered()'), self.main_widget.model_widget.add_model)
        self.connect(erase_all_audios_action, SIGNAL('triggered()'), self.erase_audios)
        self.connect(erase_all_models_action, SIGNAL('triggered()'), self.erase_models)
        self.connect(close_action, SIGNAL('triggered()'), self.close)
        self.connect(about_action, SIGNAL('triggered()'), self.about)

        # Actions to menus
        self.menu_bar.addMenu('&Archivo').addActions([add_audio_action, add_model_action, erase_all_audios_action, erase_all_models_action, close_action])
        self.menu_bar.addMenu('A&cerca').addActions([about_action])

        self.setCentralWidget(self.main_widget)

        self.setMinimumWidth(800)
        self.setMinimumHeight(480)

    def erase_audios(self):
        self.main_widget.audio_widget.audios.selectAll()
        self.main_widget.audio_widget.erase_audio()

    def erase_models(self):
        self.main_widget.model_widget.models.selectAll()
        self.main_widget.model_widget.erase_model()

    def about(self):
        about = AboutDlg(self)
        about.exec_()


class MainWidget(QWidget):

    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)
        state.main_widget = self

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.do = QPushButton('Reconocer')

        self.audio_widget = AudiosWidget()
        self.model_widget = ModelsWidget()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.audio_widget)
        splitter.addWidget(self.model_widget)

        space_height_label = QLabel()
        space_height_label.setFixedHeight(10)
        space_width_label = QLabel()
        space_width_label.setMinimumWidth(33)

        do_layout = QHBoxLayout()
        do_layout.addWidget(self.progress_bar)
        do_layout.addWidget(space_width_label)
        do_layout.addWidget(self.do)

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(space_height_label)
        layout.addLayout(do_layout)

        self.setLayout(layout)

        # Connections
        self.connect(self.audio_widget.train, SIGNAL("clicked()"), self.change_layout)
        self.connect(self.do, SIGNAL("clicked()"), self.do_your_thing)

    def do_your_thing(self):
        if self.do.text() == 'Cancelar':
            binding.cancel_process()
            WarningDlg(unicode('Proceso cancelado', encoding='cp850'), parent=self).exec_()

            if self.audio_widget.train.isChecked():
                self.do.setText('Entrenar...')
            else:
                self.do.setText('Reconocer')

        else:
            if self.audio_widget.train.isChecked():
                if len(state.audios) == 0:
                    WarningDlg('Elija al menos un archivo de audio para el entrenamiento', parent=self).exec_()
                else:
                    hmm_options = OptionDlg(state, parent=self)
                    while hmm_options.exec_():
                        address = str(hmm_options.save_line_edit.text())
                        if not os.path.exists(os.path.dirname(address)):
                            WarningDlg(unicode('El directorio no existe', encoding='cp850'), parent=self).exec_()
                            continue
                        if os.path.basename(address).split('.')[-1] != 'rdx' or len(address) <= 4:
                            WarningDlg(unicode('Nombre de fichero inv\xa0lido. Utilice "Examinar..."', encoding='cp850'),
                                       parent=self).exec_()
                            continue
                        else:
                            self.do.setText('Cancelar')

                            state.last_save_dir = str(address)
                            binding.set_state(state)
                            binding.train()
                            break
            else:
                if len(state.audios) == 0:
                    WarningDlg('No existen archivos de audio para reconocer', parent=self).exec_()
                elif len(state.models) == 0:
                    WarningDlg('No existen modelos entrenados', parent=self).exec_()
                else:
                    self.do.setText('Cancelar')

                    for index in range(self.model_widget.models.rowCount()):
                        self.model_widget.models.setItem(index, 1, QTableWidgetItem('0'))
                        self.model_widget.models.setItem(index, 2, QTableWidgetItem('0%'))

                    binding.set_state(state)
                    binding.recognize()

    def change_layout(self):
        if self.sender().isChecked():
            self.model_widget.setVisible(False)
            self.do.setText('Entrenar...')
        else:
            self.model_widget.setVisible(True)
            self.do.setText('Reconocer')


class AudiosWidget(QWidget):

    def __init__(self, checked=False, parent=None):
        super(AudiosWidget, self).__init__(parent)
        audio_label = QLabel('Audios importados:')
        self.audio_label = audio_label

        self.train = QCheckBox('Usar audios para entrenar nuevo modelo')
        self.train.setChecked(checked)

        self.audios_erase = QPushButton('Borrar')
        self.audios_add = QPushButton(unicode('A\xa4adir...', encoding='cp850'))

        audios_buttons_layout = QHBoxLayout()
        audios_buttons_layout.addWidget(self.train)
        audios_buttons_layout.addStretch()
        audios_buttons_layout.addWidget(self.audios_erase)
        audios_buttons_layout.addWidget(self.audios_add)

        self.audios = QTableWidget()
        self.audios.setAlternatingRowColors(True)
        self.audios.setEditTriggers(QTableWidget.NoEditTriggers)
        self.audios.setSelectionBehavior(QTableWidget.SelectRows)
        self.audios.setSelectionMode(QTableWidget.MultiSelection)
        self.audios.setColumnCount(2)
        self.audios.setHorizontalHeaderLabels(['Nombre', 'Modelo'])

        audios_layout = QVBoxLayout()
        audios_layout.addWidget(audio_label)
        audios_layout.addWidget(self.audios)
        audios_layout.addLayout(audios_buttons_layout)
        audios_layout.setMargin(0)
        self.setLayout(audios_layout)

        # Connections
        self.connect(self.audios_add, SIGNAL("clicked()"), self.add_audio)
        self.connect(self.audios_erase, SIGNAL("clicked()"), self.erase_audio)

    def add_audio(self):
        files = QFileDialog.getOpenFileNames(self, 'Elija los archivos de audio...', state.last_audio_dir, 'Archivos de audio (*.wav)')

        # Add the new audios at the end
        item_index = self.audios.rowCount()
        for count, audio_address in enumerate(files):
            # Save the audio address for training or recognizing
            state.audios.append(str(audio_address))
            if count == 0:
                # Saving the last directory used for later initialization
                state.last_audio_dir = os.path.dirname(state.audios[-1])

            item_name = QTableWidgetItem(os.path.basename(state.audios[-1]))
            self.audios.setRowCount(self.audios.rowCount() + 1)
            self.audios.setItem(item_index, 0, item_name)
            item_index += 1

    def erase_audio(self):
        to_remove = self.audios.selectedItems()
        for i in to_remove:
            state.audios.pop(i.row())
            self.audios.removeRow(i.row())


class ModelsWidget(QWidget):

    def __init__(self, parent=None):
        super(ModelsWidget, self).__init__(parent)
        recog_label = QLabel('Modelos importados:')

        self.models = QTableWidget()
        self.models.setAlternatingRowColors(True)
        self.models.setEditTriggers(QTableWidget.NoEditTriggers)
        self.models.setSelectionBehavior(QTableWidget.SelectRows)
        self.models.setSelectionMode(QTableWidget.MultiSelection)
        self.models.setColumnCount(3)
        self.models.setHorizontalHeaderLabels(['Nombre', 'Audios', 'Porcentaje'])

        self.models_erase = QPushButton('Borrar')
        self.models_add = QPushButton(unicode('A\xa4adir...', encoding='cp850'))

        models_buttons_layout = QHBoxLayout()
        models_buttons_layout.addStretch()
        models_buttons_layout.addWidget(self.models_erase)
        models_buttons_layout.addWidget(self.models_add)

        models_layout = QVBoxLayout()
        models_layout.addWidget(recog_label)
        models_layout.addWidget(self.models)
        models_layout.addLayout(models_buttons_layout)
        models_layout.setMargin(0)
        self.setLayout(models_layout)

        # Connections
        self.connect(self.models_add, SIGNAL("clicked()"), self.add_model)
        self.connect(self.models_erase, SIGNAL("clicked()"), self.erase_model)

    def add_model(self):
        files = QFileDialog.getOpenFileNames(self, 'Elija los modelos...', state.last_model_dir, 'Archivos de reconocimiento (*.rdx)')

        # Add the new model at the end
        item_index = self.models.rowCount()
        for count, model_address in enumerate(files):

            # Save the model address for training or recognizing
            state.models.append(str(model_address))
            if count == 0:
                # Saving the last directory used for later initialization
                state.last_model_dir = os.path.dirname(state.models[-1])

            item_name = QTableWidgetItem(os.path.basename(state.models[-1]))
            self.models.setRowCount(self.models.rowCount() + 1)
            self.models.setItem(item_index, 0, item_name)
            self.models.setItem(item_index, 1, QTableWidgetItem(str(0)))
            self.models.setItem(item_index, 2, QTableWidgetItem(str(0) + '%'))
            item_index += 1

    def erase_model(self):
        to_remove = self.models.selectedItems()
        for i in to_remove:
            state.models.pop(i.row())
            self.models.removeRow(i.row())