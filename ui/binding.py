__author__ = 'RodXander'

import os
from threading import Thread

from PyQt4.QtGui import QTableWidgetItem

from logic import hmm
from ui.dialogs import WarningDlg


state = None
vector_length = 13
action_object = None


def set_state(new_state):
    global state
    state = new_state


def process_finished():
    WarningDlg(unicode('Operaci\xa2n terminada', encoding='cp850'), parent=state.main_widget).exec_()
    state.main_widget.progress_bar.setValue(0)
    state.main_widget.do.setDisabled(False)

    if type(action_object) is hmm.RecognizeModel:
        state.main_widget.do.setText('Reconocer')
    else:
        state.main_widget.do.setText('Entrenar...')


def cancel_process():
    if action_object is not None:
        action_object.cancel = True


def set_value_recognize(value, index_audio, index_model):
    state.main_widget.progress_bar.setValue(value)

    if index_audio != -1 and index_model != -1:
        best_match = QTableWidgetItem(str(index_model + 1))
        state.main_widget.audio_widget.audios.setItem(index_audio, 1, best_match)

        models_count[index_model] += 1
        state.main_widget.model_widget.models.setItem(index_model, 1, QTableWidgetItem(str(models_count[index_model])))
        state.main_widget.model_widget.models.setItem(index_model, 2, QTableWidgetItem(str(float(models_count[index_model]) / len(state.audios) * 100) + '%'))


models_count = []   # Store the number of audios recognized by a given model


def recognize():
    global models_count, action_object

    models_count = [0] * len(state.models)

    recognize_object = hmm.RecognizeModel(state.models, state.audios)
    recognize_object.progress_signal.connect(set_value_recognize)
    recognize_object.end_proccess.connect(process_finished)

    thread = Thread(target=recognize_object.recognize)
    thread.setDaemon(True)
    thread.start()

    action_object = recognize_object

def train():
    global action_object

    initial = hmm.HMM(state.nstates, vector_length, state.ncomponents)

    training_object = hmm.TrainModel(initial, state.audios, state.last_save_dir,
                                     niters_bw=state.niterations_bw, apply_baum_welch=state.make_bw,
                                     nsegmentations=state.nsegmentations)

    training_object.progress_signal.connect(state.main_widget.progress_bar.setValue)
    training_object.end_proccess.connect(process_finished)

    thread = Thread(target=training_object.train)
    thread.setDaemon(True)
    thread.start()

    action_object = training_object


class AppState:
    def __init__(self):
        self.last_audio_dir = os.getcwd()
        self.last_model_dir = os.getcwd()
        self.last_save_dir = os.getcwd() + os.sep + 'Sin nombre.rdx'
        self.audios = []
        self.models = []

        self.thread = None
        self.main_widget = None

        # Values to feed the functions of the logic layer
        self.nstates = None
        self.ncomponents = None
        self.nsegmentations = None
        self.niterations_bw = None
        self.make_bw = None
        self.default_model_values()

    def default_model_values(self):
        self.nstates = 1
        self.ncomponents = 3
        self.nsegmentations = 150
        self.niterations_bw = 1
        self.make_bw = True
