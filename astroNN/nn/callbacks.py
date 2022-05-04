import csv
import os

import numpy as np
from tensorflow import keras as tfk

Callback = tfk.callbacks.Callback


class VirutalCSVLogger(Callback):
    """
    A modification of keras' CSVLogger, but not actually write a file until you call method to save

    :param filename: filename of the log to be saved on disk
    :type filename: str
    :param separator: separator of fields
    :type separator: str
    :param append: whether allow append or not
    :type append: bool
    :return: callback instance
    :rtype: object
    :History:
        | 2018-Feb-22 - Written - Henry Leung (University of Toronto)
        | 2018-Mar-12 - Update - Henry Leung (University of Toronto)
    """

    def __init__(self, filename="training_history.csv", separator=",", append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.csv_file = None
        self.epoch = []
        self.history = {}
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def savefile(self, folder_name=None):
        """
        the method to actually save the file to disk

        :param folder_name: foldername, can be None to save to current directory
        :type folder_name: Union[NoneType, str]
        """
        if folder_name is not None:
            full_path = os.path.normpath(os.path.join(os.getcwd(), folder_name))
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            self.filename = os.path.join(full_path, self.filename)

        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, "a")
        else:
            self.csv_file = open(self.filename, "w")

        class CustomDialect(csv.excel):
            delimiter = self.sep

        self.keys = sorted(self.history.keys())

        self.writer = csv.DictWriter(
            self.csv_file, fieldnames=["epoch"] + self.keys, dialect=CustomDialect
        )
        if self.append_header:
            self.writer.writeheader()

        for i in self.epoch:
            self.writer.writerow(
                {
                    **{"epoch": self.epoch[i]},
                    **dict([(k, self.history[k][i]) for k in self.keys]),
                }
            )
        self.csv_file.close()


class ErrorOnNaN(Callback):
    """
    Callback that raise error when a NaN is encountered.

    :return: callback instance
    :rtype: object
    :History:
        | 2018-May-07 - Written - Henry Leung (University of Toronto)
        | 2021-Apr-22 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, monitor="loss"):
        super().__init__()
        self.monitor = monitor

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        monitor = logs.get(self.monitor)
        if monitor is not None:
            if np.isnan(monitor) or np.isinf(monitor):
                self.model.stop_training = True
                raise ValueError(
                    f"Batch {int(batch)}: Invalid loss, terminating training"
                )
