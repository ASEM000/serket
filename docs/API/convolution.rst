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
    
    .. image:: ../_static/fft_bench.svg
        :width: 600
        :align: center
    

    The benchmark use ``FFTConv2D`` against ``Conv2D`` with ``in_features=3``, ``out_features=64``, and ``input_size=(10, 3, 128, 128)``


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

.. autoclass:: SpectralConv1D
.. autoclass:: SpectralConv2D
.. autoclass:: SpectralConv3D

.. autofunction:: conv_nd
.. autofunction:: depthwise_conv_nd
.. autofunction:: depthwise_fft_conv_nd
.. autofunction:: fft_conv_nd
.. autofunction:: local_conv_nd
.. autofunction:: separable_conv_nd
.. autofunction:: separable_fft_conv_nd
.. autofunction:: conv_nd_transpose
.. autofunction:: fft_conv_nd_transpose