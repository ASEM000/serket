# Copyright 2023 serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from serket._src.image.augment import (
    AdjustBrightness2D,
    AdjustContrast2D,
    AdjustHue2D,
    AdjustLog2D,
    AdjustSaturation2D,
    AdjustSigmoid2D,
    FourierDomainAdapt2D,
    Pixelate2D,
    PixelShuffle2D,
    Posterize2D,
    RandomBrightness2D,
    RandomContrast2D,
    RandomHue2D,
    RandomJigSaw2D,
    RandomSaturation2D,
    Solarize2D,
)
from serket._src.image.color import (
    GrayscaleToRGB2D,
    HSVToRGB2D,
    RGBToGrayscale2D,
    RGBToHSV2D,
)
from serket._src.image.filter import (
    AvgBlur2D,
    BilateralBlur2D,
    BlurPool2D,
    BoxBlur2D,
    ElasticTransform2D,
    FFTAvgBlur2D,
    FFTBlurPool2D,
    FFTBoxBlur2D,
    FFTElasticTransform2D,
    FFTGaussianBlur2D,
    FFTLaplacian2D,
    FFTMotionBlur2D,
    FFTSobel2D,
    FFTUnsharpMask2D,
    GaussianBlur2D,
    JointBilateralBlur2D,
    Laplacian2D,
    MedianBlur2D,
    MotionBlur2D,
    Sobel2D,
    UnsharpMask2D,
    avg_blur_2d,
    bilateral_blur_2d,
    blur_pool_2d,
    box_blur_2d,
    elastic_transform_2d,
    fft_avg_blur_2d,
    fft_blur_pool_2d,
    fft_box_blur_2d,
    fft_elastic_transform_2d,
    fft_filter_2d,
    fft_gaussian_blur_2d,
    fft_laplacian_2d,
    fft_motion_blur_2d,
    fft_sobel_2d,
    fft_unsharp_mask_2d,
    filter_2d,
    gaussian_blur_2d,
    joint_bilateral_blur_2d,
    laplacian_2d,
    median_blur_2d,
    motion_blur_2d,
    sobel_2d,
    unsharp_mask_2d,
)
from serket._src.image.geometric import (
    HorizontalFlip2D,
    HorizontalShear2D,
    HorizontalTranslate2D,
    RandomHorizontalFlip2D,
    RandomHorizontalShear2D,
    RandomHorizontalTranslate2D,
    RandomPerspective2D,
    RandomRotate2D,
    RandomVerticalFlip2D,
    RandomVerticalShear2D,
    RandomVerticalTranslate2D,
    RandomWaveTransform2D,
    Rotate2D,
    VerticalFlip2D,
    VerticalShear2D,
    VerticalTranslate2D,
    WaveTransform2D,
)

__all__ = [
    # augment
    "AdjustBrightness2D",
    "AdjustContrast2D",
    "AdjustHue2D",
    "AdjustLog2D",
    "AdjustSaturation2D",
    "AdjustSigmoid2D",
    "FourierDomainAdapt2D",
    "RandomJigSaw2D",
    "Pixelate2D",
    "PixelShuffle2D",
    "Posterize2D",
    "RandomBrightness2D",
    "RandomContrast2D",
    "RandomHue2D",
    "RandomSaturation2D",
    "Solarize2D",
    # filter
    "AvgBlur2D",
    "BilateralBlur2D",
    "BlurPool2D",
    "BoxBlur2D",
    "ElasticTransform2D",
    "FFTAvgBlur2D",
    "FFTBlurPool2D",
    "FFTBoxBlur2D",
    "FFTElasticTransform2D",
    "FFTGaussianBlur2D",
    "FFTLaplacian2D",
    "FFTMotionBlur2D",
    "FFTSobel2D",
    "FFTUnsharpMask2D",
    "GaussianBlur2D",
    "JointBilateralBlur2D",
    "Laplacian2D",
    "MedianBlur2D",
    "MotionBlur2D",
    "Sobel2D",
    "UnsharpMask2D",
    "avg_blur_2d",
    "bilateral_blur_2d",
    "blur_pool_2d",
    "box_blur_2d",
    "elastic_transform_2d",
    "fft_avg_blur_2d",
    "fft_blur_pool_2d",
    "fft_box_blur_2d",
    "fft_elastic_transform_2d",
    "fft_gaussian_blur_2d",
    "fft_laplacian_2d",
    "fft_motion_blur_2d",
    "fft_sobel_2d",
    "fft_unsharp_mask_2d",
    "gaussian_blur_2d",
    "joint_bilateral_blur_2d",
    "laplacian_2d",
    "median_blur_2d",
    "motion_blur_2d",
    "sobel_2d",
    "unsharp_mask_2d",
    "filter_2d",
    "fft_filter_2d",
    # geometric
    "CenterCrop2D",
    "HorizontalFlip2D",
    "HorizontalShear2D",
    "HorizontalTranslate2D",
    "RandomHorizontalFlip2D",
    "RandomHorizontalShear2D",
    "RandomHorizontalTranslate2D",
    "RandomPerspective2D",
    "RandomRotate2D",
    "RandomVerticalFlip2D",
    "RandomVerticalShear2D",
    "RandomVerticalTranslate2D",
    "RandomWaveTransform2D",
    "Rotate2D",
    "VerticalFlip2D",
    "VerticalShear2D",
    "VerticalTranslate2D",
    "WaveTransform2D",
    # color
    "GrayscaleToRGB2D",
    "HSVToRGB2D",
    "RGBToGrayscale2D",
    "RGBToHSV2D",
]
