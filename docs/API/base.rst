
``Serket`` NN API 
======================


Fully connected
---------------------------------
.. currentmodule:: serket.nn

.. autoclass:: FNN
.. autoclass:: MLP

Linear
---------------------------------
.. currentmodule:: serket.nn
    
.. autoclass:: Linear
.. autoclass:: Bilinear
.. autoclass:: Identity
.. autoclass:: Multilinear
.. autoclass:: GeneralLinear
.. autoclass:: Embedding

Dropout
---------------------------------
.. currentmodule:: serket.nn
    
.. autoclass:: Dropout
.. autoclass:: Dropout1D
.. autoclass:: Dropout2D
.. autoclass:: Dropout3D


Containers
---------------------------------
.. currentmodule:: serket.nn
    

.. autoclass:: Sequential

Pooling
---------------------------------
.. currentmodule:: serket.nn
    

.. autoclass:: MaxPool1D
.. autoclass:: MaxPool2D
.. autoclass:: MaxPool3D
.. autoclass:: AvgPool1D
.. autoclass:: AvgPool2D
.. autoclass:: AvgPool3D
.. autoclass:: GlobalAvgPool1D
.. autoclass:: GlobalAvgPool2D
.. autoclass:: GlobalAvgPool3D
.. autoclass:: GlobalMaxPool1D
.. autoclass:: GlobalMaxPool2D
.. autoclass:: GlobalMaxPool3D
.. autoclass:: LPPool1D
.. autoclass:: LPPool2D
.. autoclass:: LPPool3D
.. autoclass:: AdaptiveAvgPool1D
.. autoclass:: AdaptiveAvgPool2D
.. autoclass:: AdaptiveAvgPool3D
.. autoclass:: AdaptiveMaxPool1D
.. autoclass:: AdaptiveMaxPool2D
.. autoclass:: AdaptiveMaxPool3D


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

Normalization
---------------------------------
.. currentmodule:: serket.nn
    
.. autoclass:: LayerNorm
.. autoclass:: InstanceNorm
.. autoclass:: GroupNorm
.. autoclass:: BatchNorm

Image filtering
---------------------------------
.. currentmodule:: serket.nn
    
.. autoclass:: AvgBlur2D
.. autoclass:: GaussianBlur2D
.. autoclass:: Filter2D
.. autoclass:: FFTFilter2D

Misc
---------------------------------
.. currentmodule:: serket.nn
    
.. autoclass:: FlipLeftRight2D
.. autoclass:: FlipUpDown2D
.. autoclass:: Resize1D
.. autoclass:: Resize2D
.. autoclass:: Resize3D
.. autoclass:: Upsample1D
.. autoclass:: Upsample2D
.. autoclass:: Upsample3D
.. autoclass:: Pad1D
.. autoclass:: Pad2D
.. autoclass:: Pad3D

.. autoclass:: VGG16Block
.. autoclass:: VGG19Block
.. autoclass:: UNetBlock

.. autoclass:: Crop1D
.. autoclass:: Crop2D
.. autoclass:: Crop3D

.. autoclass:: Flatten
.. autoclass:: Unflatten

.. autoclass:: HistogramEqualization2D
.. autoclass:: PixelShuffle2D

Random transforms
---------------------------------
.. currentmodule:: serket.nn
    
.. autoclass:: RandomCrop1D
.. autoclass:: RandomCrop2D
.. autoclass:: RandomCrop3D
.. autoclass:: RandomCutout1D
.. autoclass:: RandomCutout2D
.. autoclass:: RandomZoom2D
.. autoclass:: RandomApply


Activations
---------------------------------
.. currentmodule:: serket.nn
    
.. autoclass:: AdaptiveLeakyReLU
.. autoclass:: AdaptiveReLU
.. autoclass:: AdaptiveSigmoid
.. autoclass:: AdaptiveTanh
.. autoclass:: CeLU
.. autoclass:: ELU
.. autoclass:: GELU
.. autoclass:: GLU
.. autoclass:: HardShrink
.. autoclass:: HardSigmoid
.. autoclass:: HardSwish
.. autoclass:: HardTanh
.. autoclass:: LeakyReLU
.. autoclass:: LogSigmoid
.. autoclass:: LogSoftmax
.. autoclass:: Mish
.. autoclass:: PReLU
.. autoclass:: ReLU
.. autoclass:: ReLU6
.. autoclass:: SeLU
.. autoclass:: Sigmoid
.. autoclass:: SoftPlus
.. autoclass:: SoftShrink
.. autoclass:: SoftSign
.. autoclass:: SquarePlus
.. autoclass:: Swish
.. autoclass:: Snake
.. autoclass:: Tanh
.. autoclass:: TanhShrink
.. autoclass:: ThresholdedReLU

.. autoclass:: AdjustContrast2D
.. autoclass:: RandomContrast2D

Recurrent
---------------------------------

.. currentmodule:: serket.nn
    
.. autoclass:: LSTMCell
.. autoclass:: GRUCell
.. autoclass:: SimpleRNNCell
.. autoclass:: DenseCell
.. autoclass:: ConvLSTM1DCell
.. autoclass:: ConvLSTM2DCell
.. autoclass:: ConvLSTM3DCell
.. autoclass:: ConvGRU1DCell
.. autoclass:: ConvGRU2DCell
.. autoclass:: ConvGRU3DCell
.. autoclass:: ScanRNN