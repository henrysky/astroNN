
Knwon Issues
---------------------------------------


Keras's fit_generator won't work even though astroNN's generator is made thread safe when use_multiprocessing=True
=====================================================================================================================

It is a known issue on Windows caused by python. Probably will work on Linux/MacOS.

So far the only issue is CPU can't generate data fast enough for a fast GPU (GTX970 or above and at least 4 threads CPU).

Only neccessary when you are using BCNN with GPU training

H5Loader loading the whole h5 files is a problematic approach and will eventually causes memory issue for a larger 2D dataset
===============================================================================================================================

Will be fixed in a near future

