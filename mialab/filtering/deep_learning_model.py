"""
Deep learning components for brain tissue segmentation.

This module defines:
    - BrainImageDataset
    - create_dynunet
    - train_deep_learning_model
    - infer_full_volume
    - run_deep_learning_pipeline

All forest-related stuff stays in pipeline.py.
"""

import os
import datetime
import timeit
import random  # Added for data splitting

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # Added for plotting

import SimpleITK as sitk

from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss

import pymia.evaluation.writer as writer

import mialab.data.structure as structure
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil
import mialab.filtering.preprocessing as fltr_prep


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

LOADING_KEYS = [
    structure.BrainImageTypes.T1w,
    structure.BrainImageTypes.T2w,
    structure.BrainImageTypes.GroundTruth,
    structure.BrainImageTypes.BrainMask,
    structure.BrainImageTypes.RegistrationTransform,
]


class BrainImageDataset(Dataset):
    """
    A PyTorch Dataset for loading and pre-processing 3D brain images
    and extracting patches for U-Net training.
    """

    def __init__(self, patient_data: dict, patch_size=(96, 96, 96),
     normalization_method='z_score'):
        # patient_data is crawler.data from FileSystemDataCrawler
        self.patient_data = list(patient_data.items())
        self.patch_size = patch_size
        self.normalization_method = normalization_method

        # atlas images must be loaded via putil.load_atlas_images(...)
        if not hasattr(putil, "atlas_t1") or putil.atlas_t1 is None:
            raise RuntimeError("Atlas images not loaded. Call putil.load_atlas_images(...) first.")
        self.atlas_t1 = putil.atlas_t1
        self.atlas_t2 = putil.atlas_t2

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        id_, paths_original = self.patient_data[idx]
        # work on a copy to avoid mutating crawler.data
        paths = dict(paths_original)

        # ----- 1) Load images -----
        t1_path = paths.get(structure.BrainImageTypes.T1w, "")
        t2_path = paths.get(structure.BrainImageTypes.T2w, "")
        gt_path = paths.get(structure.BrainImageTypes.GroundTruth, "")
        mask_path = paths.get(structure.BrainImageTypes.BrainMask, "")
        transform_path = paths.get(structure.BrainImageTypes.RegistrationTransform, "")

        img_dict = {
            structure.BrainImageTypes.T1w: sitk.ReadImage(t1_path),
            structure.BrainImageTypes.T2w: sitk.ReadImage(t2_path),
            structure.BrainImageTypes.GroundTruth: sitk.ReadImage(gt_path),
            structure.BrainImageTypes.BrainMask: sitk.ReadImage(mask_path),
        }
        transform = sitk.ReadTransform(transform_path)
        img = structure.BrainImage(id_, t1_path, img_dict, transform)

        # ----- 2) Registration & resampling -----

        # T1w -> atlas
        reg_params_t1 = fltr_prep.ImageRegistrationParameters(self.atlas_t1, img.transformation)
        t1w_registered = fltr_prep.ImageRegistration().execute(
            img.images[structure.BrainImageTypes.T1w], reg_params_t1
        )

        # BrainMask -> atlas (NN)
        reg_params_mask = fltr_prep.ImageRegistrationParameters(
            self.atlas_t1, img.transformation, True
        )
        mask_registered_to_atlas = fltr_prep.ImageRegistration().execute(
            img.images[structure.BrainImageTypes.BrainMask], reg_params_mask
        )
        resampler_mask = sitk.ResampleImageFilter()
        resampler_mask.SetReferenceImage(t1w_registered)
        resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler_mask.SetDefaultPixelValue(0)
        mask_registered = resampler_mask.Execute(mask_registered_to_atlas)

        # T2w -> atlas (linear)
        reg_params_t2 = fltr_prep.ImageRegistrationParameters(self.atlas_t1, img.transformation)
        t2w_registered_to_atlas = fltr_prep.ImageRegistration().execute(
            img.images[structure.BrainImageTypes.T2w], reg_params_t2
        )
        resampler_t2 = sitk.ResampleImageFilter()
        resampler_t2.SetReferenceImage(t1w_registered)
        resampler_t2.SetInterpolator(sitk.sitkLinear)
        resampler_t2.SetDefaultPixelValue(0)
        t2w_registered = resampler_t2.Execute(t2w_registered_to_atlas)

        # GroundTruth -> atlas (NN)
        gt_registered_to_atlas = fltr_prep.ImageRegistration().execute(
            img.images[structure.BrainImageTypes.GroundTruth], reg_params_mask
        )
        resampler_gt = sitk.ResampleImageFilter()
        resampler_gt.SetReferenceImage(t1w_registered)
        resampler_gt.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler_gt.SetDefaultPixelValue(0)
        gt_registered = resampler_gt.Execute(gt_registered_to_atlas)

        # ----- 3) Skull-strip & normalize -----
        skullstrip_params = fltr_prep.SkullStrippingParameters(mask_registered)
        normalizer = get_normalizer(self.normalization_method)

        # Apply Skull Stripping
        skullstripped_t1 = fltr_prep.SkullStripping().execute(t1w_registered, skullstrip_params)
        skullstripped_t2 = fltr_prep.SkullStripping().execute(t2w_registered, skullstrip_params)
        skullstripped_gt = fltr_prep.SkullStripping().execute(gt_registered, skullstrip_params)

        # Prepare parameters for Normalization
        # ZScore/MinMax/Percentile ignore these params, but HistogramMatching needs them.
        param_t1 = fltr_prep.NormalizationParameters(self.atlas_t1)
        param_t2 = fltr_prep.NormalizationParameters(self.atlas_t2)

        # Apply Normalization
        # Note: We pass param_t1/param_t2. The execute method of ZScore/MinMax ignores it,
        # but HistogramMatching uses it.
        if self.normalization_method == 'none':
            # Skip normalization, just pass the skullstripped images
            normalized_t1 = skullstripped_t1
            normalized_t2 = skullstripped_t2
        else:
            # Use the normalizer object as before
            normalized_t1 = normalizer.execute(skullstripped_t1, param_t1)
            normalized_t2 = normalizer.execute(skullstripped_t2, param_t2)

        # ----- 4) Convert to numpy / tensors -----
        t1w_np = sitk.GetArrayFromImage(normalized_t1).astype(np.float32)
        t2w_np = sitk.GetArrayFromImage(normalized_t2).astype(np.float32)
        gt_np = sitk.GetArrayFromImage(skullstripped_gt).astype(np.int64)

        # add channel dim
        t1w_np = np.expand_dims(t1w_np, axis=0)
        t2w_np = np.expand_dims(t2w_np, axis=0)
        gt_np = np.expand_dims(gt_np, axis=0)

        # input = (2, D, H, W)
        input_image_np = np.concatenate([t1w_np, t2w_np], axis=0)
        input_tensor = torch.from_numpy(input_image_np)
        label_tensor = torch.from_numpy(gt_np).squeeze(0)  # (D, H, W)

        # ----- 5) Extract random patch -----
        non_bg_indices = np.argwhere(label_tensor.numpy() > 0)
        z_dim, y_dim, x_dim = input_tensor.shape[1:]

        if len(non_bg_indices) == 0:
            # random center if no foreground
            cz = np.random.randint(
                self.patch_size[0] // 2,
                max(self.patch_size[0] // 2 + 1, z_dim - self.patch_size[0] // 2),
            )
            cy = np.random.randint(
                self.patch_size[1] // 2,
                max(self.patch_size[1] // 2 + 1, y_dim - self.patch_size[1] // 2),
            )
            cx = np.random.randint(
                self.patch_size[2] // 2,
                max(self.patch_size[2] // 2 + 1, x_dim - self.patch_size[2] // 2),
            )
        else:
            center_idx = np.random.randint(0, len(non_bg_indices))
            cz, cy, cx = non_bg_indices[center_idx]
            cz = max(self.patch_size[0] // 2, min(cz, z_dim - self.patch_size[0] // 2))
            cy = max(self.patch_size[1] // 2, min(cy, y_dim - self.patch_size[1] // 2))
            cx = max(self.patch_size[2] // 2, min(cx, x_dim - self.patch_size[2] // 2))

        z_slice = slice(cz - self.patch_size[0] // 2, cz + self.patch_size[0] // 2)
        y_slice = slice(cy - self.patch_size[1] // 2, cy + self.patch_size[1] // 2)
        x_slice = slice(cx - self.patch_size[2] // 2, cx + self.patch_size[2] // 2)

        image_patch = input_tensor[:, z_slice, y_slice, x_slice]
        label_patch = label_tensor[z_slice, y_slice, x_slice]

        # pad if patch is smaller than desired
        if image_patch.shape[1:] != self.patch_size or label_patch.shape != self.patch_size:
            pad_image = torch.zeros(input_tensor.shape[0], *self.patch_size)
            pad_label = torch.zeros(self.patch_size, dtype=label_patch.dtype)
            pad_image[:, : image_patch.shape[1], : image_patch.shape[2], : image_patch.shape[3]] = image_patch
            pad_label[: label_patch.shape[0], : label_patch.shape[1], : label_patch.shape[2]] = label_patch
            image_patch = pad_image
            label_patch = pad_label

        return image_patch, label_patch


# -------------------------------------------------------------------------
# Model + training
# -------------------------------------------------------------------------


def create_dynunet(in_channels: int = 2, out_channels: int = 6) -> DynUNet:
    """Create a 3D DynUNet with fixed hyperparameters."""
    model = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[[3, 3, 3]] * 5,
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=[16, 32, 64, 128, 256],
        deep_supervision=False,
    )
    return model


def train_deep_learning_model(
    train_loader: DataLoader,
    val_loader: DataLoader,  # Added validation loader
    device: torch.device,
    num_epochs: int = 1,
    in_channels: int = 2,
    out_channels: int = 6,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
) -> tuple:
    """Train DynUNet on patches from train_loader."""
    model = create_dynunet(in_channels=in_channels, out_channels=out_channels).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # History tracking
    train_loss_history = []
    val_loss_history = []

    start_time = timeit.default_timer()
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            labels = labels.long().unsqueeze(1)  # shape (B, 1, D, H, W)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(step, 1)
        train_loss_history.append(epoch_loss)

        # --- Validation ---
        model.eval()
        epoch_val_loss = 0.0
        val_step = 0
        with torch.no_grad():
            for batch_data in val_loader:
                val_step += 1
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                labels = labels.long().unsqueeze(1)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= max(val_step, 1)
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    print("Training Time elapsed:", timeit.default_timer() - start_time, "s")
    
    return model, train_loss_history, val_loss_history


# -------------------------------------------------------------------------
# Inference helper
# -------------------------------------------------------------------------


def infer_full_volume(
    model: torch.nn.Module,
    x: torch.Tensor,
    roi_size,
    device: torch.device,
    overlap: float = 0.25,
) -> np.ndarray:
    """
    Run sliding-window inference on a single 3D volume.
    x: tensor of shape (1, C, D, H, W)
    returns np.ndarray of shape (D, H, W), dtype=uint8
    """
    model.eval()

    if x.device != device:
        x = x.to(device)

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
        )

    pred_np = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_np

class NoNormalization:
    def execute(self, image, params=None):
        return image  # <--- You must have this, not 'pass'


def get_normalizer(method: str):
    """Factory to get the normalization filter based on string name."""
    if method == 'z_score':
        return fltr_prep.ZScore()
    elif method == 'min_max':
        return fltr_prep.MinMax()
    elif method == 'percentile':
        return fltr_prep.Percentile()
    elif method == 'histogram_matching': # FIX: Correctly returns the HistogramMatching instance
        # Default params from your class definition
        return fltr_prep.HistogramMatching()
    else:
        # Note: 'none' case is handled by the caller in this implementation, 
        # but if it was passed here, we can return the default.
        return fltr_prep.ImageNormalization()

# -------------------------------------------------------------------------
# Full deep pipeline (training + testing)
# -------------------------------------------------------------------------


def run_deep_learning_pipeline(
    result_dir: str,
    data_atlas_dir: str,
    data_train_dir: str,
    data_test_dir: str,
    num_epochs: int = 80,
    patch_size=(128, 128, 128),
    batch_size: int = 2,
    normalization_method: str = 'z_score'
) -> None:
    """
    Full deep-learning pipeline:
        - load atlas
        - train DynUNet on training set
        - run full-volume inference on test set
        - write evaluation CSVs into result_dir/<timestamp>
    """
    # ----- Load atlas -----
    putil.load_atlas_images(data_atlas_dir)

    # ----- Training -----
    print("-" * 5, "Training (deep)...")

    crawler_train = futil.FileSystemDataCrawler(
        data_train_dir,
        LOADING_KEYS,
        futil.BrainImageFilePathGenerator(),
        futil.DataDirectoryFilter(),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Split data into train and validation (80/20)
    data_items = list(crawler_train.data.items())
    random.seed(42)
    random.shuffle(data_items)
    
    split_index = int(len(data_items) * 0.8)
    train_data = dict(data_items[:split_index])
    val_data = dict(data_items[split_index:])
    
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Create Datasets and Loaders
    train_dataset = BrainImageDataset(train_data, patch_size=patch_size, normalization_method=normalization_method)
    val_dataset = BrainImageDataset(val_data, patch_size=patch_size, normalization_method=normalization_method)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # Validation usually doesn't need shuffle, but patches are random, so shuffle helps variety
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model, train_losses, val_losses = train_deep_learning_model(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        in_channels=2,
        out_channels=6,
    )

    # timestamped result directory
    t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_dir = os.path.join(result_dir, t + normalization_method )
    os.makedirs(result_dir, exist_ok=True)

    # Plot Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve - Normalization: {normalization_method}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(result_dir, 'learning_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Learning curve saved to {plot_path}")

    # save model
    model_path = os.path.join(result_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ----- Testing -----
    print("-" * 5, "Testing (deep)...")

    evaluator = putil.init_evaluator()

    crawler_test = futil.FileSystemDataCrawler(
        data_test_dir,
        LOADING_KEYS,
        futil.BrainImageFilePathGenerator(),
        futil.DataDirectoryFilter(),
    )

    for sid, original_paths in crawler_test.data.items():
        print("-" * 10, "Testing", sid)
        paths = dict(original_paths)

        t1_path = paths.get(structure.BrainImageTypes.T1w)
        t2_path = paths.get(structure.BrainImageTypes.T2w)
        mask_path = paths.get(structure.BrainImageTypes.BrainMask)
        transform_path = paths.get(structure.BrainImageTypes.RegistrationTransform)
        gt_path = paths.get(structure.BrainImageTypes.GroundTruth)

        if not all([t1_path, t2_path, mask_path, transform_path]):
            print(f"  Skipping {sid}: missing T1, T2, BrainMask, or Transform.")
            continue

        try:
            original_t1_img = sitk.ReadImage(t1_path)
            gt_original = sitk.ReadImage(gt_path) if gt_path is not None else None

            t1_img = sitk.ReadImage(t1_path)
            t2_img = sitk.ReadImage(t2_path)
            mask_img = sitk.ReadImage(mask_path)
            transform = sitk.ReadTransform(transform_path)
        except Exception as e:
            print(f"  Skipping {sid}: Error loading files. {e}")
            continue

        print(f"  Preprocessing {sid}...")

        # A) T1w -> atlas
        reg_params_t1 = fltr_prep.ImageRegistrationParameters(putil.atlas_t1, transform)
        t1w_registered = fltr_prep.ImageRegistration().execute(t1_img, reg_params_t1)

        # B) BrainMask -> atlas
        reg_params_mask = fltr_prep.ImageRegistrationParameters(putil.atlas_t1, transform, True)
        mask_registered_to_atlas = fltr_prep.ImageRegistration().execute(mask_img, reg_params_mask)

        resampler_mask = sitk.ResampleImageFilter()
        resampler_mask.SetReferenceImage(t1w_registered)
        resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler_mask.SetDefaultPixelValue(0)
        mask_registered = resampler_mask.Execute(mask_registered_to_atlas)

        # C) T2w -> atlas
        reg_params_t2 = fltr_prep.ImageRegistrationParameters(putil.atlas_t1, transform)
        t2w_registered_to_atlas = fltr_prep.ImageRegistration().execute(t2_img, reg_params_t2)

        resampler_t2 = sitk.ResampleImageFilter()
        resampler_t2.SetReferenceImage(t1w_registered)
        resampler_t2.SetInterpolator(sitk.sitkLinear)
        resampler_t2.SetDefaultPixelValue(0)
        t2w_registered = resampler_t2.Execute(t2w_registered_to_atlas)

        # D) Skull-strip & normalize
        skullstrip_params = fltr_prep.SkullStrippingParameters(mask_registered)
        normalizer = get_normalizer(normalization_method)

        skullstripped_t1 = fltr_prep.SkullStripping().execute(t1w_registered, skullstrip_params)
        skullstripped_t2 = fltr_prep.SkullStripping().execute(t2w_registered, skullstrip_params)

        # Create params with the Atlas as reference

        if normalization_method == 'none':
            # Skip normalization
            normalized_t1 = skullstripped_t1
            normalized_t2 = skullstripped_t2
        else:
            # Normal logic
            normalizer = get_normalizer(normalization_method)
            param_t1 = fltr_prep.NormalizationParameters(putil.atlas_t1)
            param_t2 = fltr_prep.NormalizationParameters(putil.atlas_t2)
            
            normalized_t1 = normalizer.execute(skullstripped_t1, param_t1)
            normalized_t2 = normalizer.execute(skullstripped_t2, param_t2)

        # E) Create input tensor
        t1w_np = sitk.GetArrayFromImage(normalized_t1).astype(np.float32)
        t2w_np = sitk.GetArrayFromImage(normalized_t2).astype(np.float32)
        x_np = np.stack([t1w_np, t2w_np], axis=0)

        x = torch.from_numpy(x_np).unsqueeze(0)  # (1, 2, D, H, W)

        roi_size = tuple(int(min(a, b)) for a, b in zip(t1w_np.shape, patch_size))

        print(f"  Running inference for {sid}...")
        inference_start_time = timeit.default_timer()

        pred_np = infer_full_volume(
            model=model,
            x=x,
            roi_size=roi_size,
            device=device,
            overlap=0.25,
        )

        print(
            f"  Inference finished for {sid}. "
            f"Time: {timeit.default_timer() - inference_start_time:.2f}s"
        )

        # convert prediction back to an image in atlas space
        pred_img_atlas = sitk.GetImageFromArray(pred_np)
        pred_img_atlas.CopyInformation(normalized_t1)

        # invert transform: atlas -> original space
        try:
            inverse_transform = transform.GetInverse()
        except Exception as e:
            print(f"  WARNING: Could not invert transform for {sid}. {e}")
            print("  Saving result in atlas space.")
            inverse_transform = sitk.Transform()  # identity

        resampler_back = sitk.ResampleImageFilter()
        resampler_back.SetReferenceImage(original_t1_img)
        resampler_back.SetTransform(inverse_transform)
        resampler_back.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler_back.SetDefaultPixelValue(0)

        seg_img_original_space = resampler_back.Execute(pred_img_atlas)

        if gt_original is not None:
            evaluator.evaluate(seg_img_original_space, gt_original, sid)

        out_path = os.path.join(result_dir, f"{sid}_SEG.mha")
        sitk.WriteImage(seg_img_original_space, out_path, True)
        print(f"  saved {out_path} (size: {seg_img_original_space.GetSize()})")

    # ----- Write evaluation CSVs -----
    result_file = os.path.join(result_dir, "results.csv")
    writer.CSVWriter(result_file).write(evaluator.results)

    print("\nSubject-wise results...")
    writer.ConsoleWriter().write(evaluator.results)

    result_summary_file = os.path.join(result_dir, "results_summary.csv")
    functions = {"MEAN": np.mean, "STD": np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)

    print("\nAggregated statistic results...")
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    evaluator.clear()