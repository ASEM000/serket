Convolution
---------------------------------

.. currentmodule:: serket.nn
    
.. autoclass:: Conv1D
.. autoclass:: Conv2D
.. autoclass:: Conv3D

.. autoclass:: Conv1DTranspose
.. autoclass:: Conv2DTranspose
.. autoclass:: Conv3DTranspose

.. autoclass:: DepthwiseConv1D
.. autoclass:: DepthwiseConv2D
.. autoclass:: DepthwiseConv3D

.. autoclass:: SeparableConv1D
.. autoclass:: SeparableConv2D
.. autoclass:: SeparableConv3D

.. autoclass:: Conv1DLocal
.. autoclass:: Conv2DLocal
.. autoclass:: Conv3DLocal

.. note::
    The ``fft`` convolution variant is useful in myriad of cases, specifically the ``fft`` variant could be faster for larger kernel sizes. the following figure compares the speed of both implementation for different kernel size on mac ``m1`` cpu setup.
    
    .. image:: fft_bench.svg
        :width: 600
        :align: center


.. autoclass:: FFTConv1D
.. autoclass:: FFTConv2D
.. autoclass:: FFTConv3D

.. autoclass:: DepthwiseFFTConv1D
.. autoclass:: DepthwiseFFTConv2D
.. autoclass:: DepthwiseFFTConv3D

.. autoclass:: FFTConv1DTranspose
.. autoclass:: FFTConv2DTranspose
.. autoclass:: FFTConv3DTranspose

.. autoclass:: SeparableFFTConv1D
.. autoclass:: SeparableFFTConv2D
.. autoclass:: SeparableFFTConv3D


