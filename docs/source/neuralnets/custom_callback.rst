
Custom Callbacks in astroNN
=======================================

A callback is a set of functions to be applied at given stages of the training procedure.
astroNN provides some customized callbacks which built on Keras and Tensorflow. Thus they are compatible with Keras
with Tensorflow backend. You can just treat astroNN customized callbacks as conventional Keras callbacks.

Virtual CSVLogger
---------------------------------------------

`Virutal_CSVLogger` is basically Keras's CSVLogger without Python 2 support and won't write the file to disk unless
`savefile()` method is called where Keras's CSVLogger will write to disk immediately.


`Virutal_CSVLogger` can be imported by

.. code-block:: python

    from astroNN.nn.utilities.callbacks import Virutal_CSVLogger

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here
        return model

    # Create a Virtual_CSVLogger instance first
    csvlogger = Virutal_CSVLogger()

    # Default filename is training_history.csv
    # You have to set filename first before passing to Keras
    csvlogger.filename = 'training_history.csv'

    model = keras_model()
    model.compile(....)

    model.fit(...,callbacks=[csvlogger])

    # Save the file to current directory
    csvlogger.savefile()

    # OR to save the file to other directory
    csvlogger.savefile(folder_name='some_folder')
