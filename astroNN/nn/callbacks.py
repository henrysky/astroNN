import csv
import os

from astroNN.config import keras_import_manager

keras = keras_import_manager()
Callback = keras.callbacks.Callback


class Virutal_CSVLogger(Callback):
    """
    NAME: Virutal_CSVLogger
    PURPOSE:
        A modification of keras' CSVLogger, but not actually write a file
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-22 - Written - Henry Leung (University of Toronto)
        2018-Mar-12 - Update - Henry Leung (University of Toronto)
    """
    def __init__(self, filename='training_history.csv', separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.csv_file = None
        self.epoch = []
        self.history = {}
        super(Virutal_CSVLogger, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def savefile(self, folder_name=None):
        if folder_name is not None:
            full_path = os.path.normpath(folder_name)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            self.filename = os.path.join(full_path, self.filename)

        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

        class CustomDialect(csv.excel):
            delimiter = self.sep

        self.keys = sorted(self.history.keys())

        self.writer = csv.DictWriter(self.csv_file, fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
        if self.append_header:
            self.writer.writeheader()

        for i in self.epoch:
            self.writer.writerow({**{'epoch': self.epoch[i]}, **dict([(k, self.history[k][i]) for k in self.keys])})
        self.csv_file.close()
