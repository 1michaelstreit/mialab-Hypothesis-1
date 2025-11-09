"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
from pymia.filtering.filter import FilterParams
import SimpleITK as sitk
import numpy as np

# container of parameter used in image normalization filter
class NormalizationParameters(pymia_fltr.FilterParams):

    def __init__(self,
                reference_image: sitk.Image,
                image_id: str = '',
                label='',
                ) -> None:
        super().__init__()
        self.reference_image = reference_image
        self.image_id = image_id
        self.label = label

class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """
        img_arr = sitk.GetArrayFromImage(image)
        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

class ZScore(ImageNormalization):

    # run constructors
    def __init__(self):
        super().__init__()

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print('Normalization: Z-Score Method')  
        img_arr = sitk.GetArrayFromImage(image)

        # Create a mask for brain tissue (non-zero voxels after skull stripping)
        brain_mask = img_arr != 0
        brain_voxels = img_arr[brain_mask]

        # If there are no brain voxels or they all have the same value, normalization is not possible/needed.
        if brain_voxels.size == 0:
            return image

        mean = np.mean(brain_voxels)
        std = np.std(brain_voxels)

        if std < 1e-6:  # Use a small epsilon for floating point comparison to avoid division by zero
            return image

        # Normalize the image using numpy. Create a new float array for the result.
        normalized_arr = np.zeros_like(img_arr, dtype=np.float32)

        # Apply z-score normalization only to the brain voxels
        normalized_arr[brain_mask] = (brain_voxels - mean) / std

        img_out = sitk.GetImageFromArray(normalized_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)

class MinMax(ImageNormalization):
    
    def __init__(self):
        super().__init__()

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print('Normalization: Min-Max Method')

        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)

        min_val = np.min(img_arr)
        max_val = np.max(img_arr)

        if max_val > min_val:
            img_arr = (img_arr - min_val) / (max_val - min_val)
        else:
            img_arr[:] = 0.0

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out
    
class Percentile(ImageNormalization):
    
    def __init__(self, lower: float = 2.0, upper: float = 98.0):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        print('Normalization: Percentile Method')

        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)

        p_low = np.percentile(img_arr, self.lower)
        p_high = np.percentile(img_arr, self.upper)

        img_arr = np.clip(img_arr, p_low, p_high)

        # Normalize to [0, 1]
        if p_high > p_low:
            img_arr = (img_arr - p_low) / (p_high - p_low)
        else:
            img_arr[:] = 0.0

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out
class HistogramMatching(ImageNormalization):
    
    def __init__(self, num_histogram_levels: int = 1024, num_match_points: int = 1000, threshold_at_mean_intensity: bool = True):
        super().__init__()
        self.num_histogram_levels = num_histogram_levels
        self.num_match_points = num_match_points
        self.threshold_at_mean_intensity = threshold_at_mean_intensity

    def execute(self, image: sitk.Image, params: NormalizationParameters = None) -> sitk.Image:
        print('Normalization: Histogram Matching Method')

        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(self.num_histogram_levels)
        matcher.SetNumberOfMatchPoints(self.num_match_points)
        if self.threshold_at_mean_intensity:
            matcher.ThresholdAtMeanIntensityOn()
        else:
            matcher.ThresholdAtMeanIntensityOff()
            
        img_out = matcher.Execute(image, params.reference_image)

        return img_out
class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The skull-stripped image.
        """
        mask = params.img_mask

        # Cast the mask to the image's pixel type
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(image.GetPixelID())
        mask = caster.Execute(mask)

        # Multiply the image with the mask to remove the skull
        return sitk.Multiply(image, mask)

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """
        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth

        # Use the ResampleImageFilter to apply the transformation
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(atlas)  # Set the output space to match the atlas
        resampler.SetTransform(transform)

        # Choose the interpolator based on the image type
        if is_ground_truth:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)

        resampler.SetDefaultPixelValue(0) # Pixels outside the moving image are set to 0

        return resampler.Execute(image)


    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)
