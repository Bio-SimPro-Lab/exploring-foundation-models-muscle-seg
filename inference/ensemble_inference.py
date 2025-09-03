import os
import numpy as np
import nibabel
import torch
import torchio as tio
import json
from argparse import Namespace
from PIL import Image
from torchvision import transforms
import scipy.special
from pathlib import Path
from models.sam import sam_model_registry
from models.sam_LoRa import LoRA_Sam
from utils.plot_utils import plot_random_slice # Import the plotting utility
from utils.plot_utils import plot_uncertainty # Import the plotting utility

def load_nifti_file(file_path):
    """Loads data from a NIfTI file."""
    return nibabel.load(file_path).get_fdata()

class SAMEnsembleTester:
    """
    Handles inference for an ensemble of SAM models (e.g., from different folds).
    Averages predictions and calculates variance.
    """
    def __init__(self, images_ts_path, images_gt_path,
                 severity_case, args_json_paths, save_base_path):
        """
        Initializes the SAMEnsembleTester.

        Args:
            weights_type (str): Type of weights ('sam' or 'medsam').
            images_ts_path (str): Path to the test images directory.
            images_gt_path (str): Path to the ground truth masks directory.
            severity_case (str): Severity level (e.g., 'healthy', 'moderate', 'severe').
            args_json_paths (list): List of paths to argument JSON files for each model in the ensemble.
            save_base_path (str): Base directory to save predictions and examples.
        """
        self.images_ts_path = images_ts_path
        self.images_gt_path = images_gt_path
        self.severity_case = severity_case
        self.args_json_paths = args_json_paths
        self.save_base_path = save_base_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.row_labels = [
            "Background", "Vastus Lateralis", "Vastus Medialis",
            "Vastus Intermedius", "Rectus Femoris", "Sartorius",
            "Gracilis", "Adductor Magnus", "Semimembranosus",
            "Semitendinosus", "Biceps Femoris Long", "Biceps Femoris Short",
            "Adductor Longus"
        ]
        self.num_classes = len(self.row_labels)

    def _load_model(self, args):
        """Loads a single SAM model from a given set of args."""
        if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla':
            model = sam_model_registry[args.arch](args,
                                                    checkpoint=os.path.join(os.getcwd(), args.dir_checkpoint, 'checkpoint_best.pth'),
                                                    num_classes=args.num_cls)
        elif args.finetune_type == 'lora':
            sam_base = sam_model_registry[args.arch](args, checkpoint=os.path.join(args.sam_ckpt),
                                                     num_classes=args.num_cls)
            model = LoRA_Sam(args, sam_base, r=4).to(self.device).sam
            model.load_state_dict(torch.load(os.path.join(args.dir_checkpoint, 'checkpoint_best.pth')), strict=False)
        return model.to(self.device).eval()

    def _preprocess_image(self, img_in, img_out, normalize):
        """Preprocesses a 2D image slice for model input."""
        img_in_scale = np.array((img_in - img_in.min()) / (img_in.max() - img_in.min() + 1e-8) * 255, dtype=np.uint8)
        img_out_scale = np.array((img_out - img_out.min()) / (img_out.max() - img_out.min() + 1e-8) * 255, dtype=np.uint8)

        img_mean = np.mean([img_in_scale, img_out_scale], axis=0).astype('uint8')
        tensor_img = np.stack((img_in_scale, img_out_scale, img_mean), axis=-1)

        pil_img = Image.fromarray(tensor_img, 'RGB')
        resized_img = transforms.Resize((1024, 1024))(pil_img)
        transformed_img = transforms.ToTensor()(resized_img)

        if normalize == 'sam':
            return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transformed_img)
        elif normalize == 'medsam':
            return transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x)))(transformed_img)
        return transformed_img

    def predict_ensemble(self):
        """
        Performs ensemble prediction on all volumes in the test set.
        Averages the softmax probabilities from multiple models.
        """
        volume_filenames = sorted(os.listdir(self.images_ts_path))
        gt_filenames = sorted(os.listdir(self.images_gt_path))
        npz_counter = 0

        for i in range(0, len(volume_filenames), 2):
            volume_path_in = os.path.join(self.images_ts_path, volume_filenames[i])
            volume_path_out = os.path.join(self.images_ts_path, volume_filenames[i+1])

            image_in_vol = tio.ScalarImage(volume_path_in)
            image_out_vol = tio.ScalarImage(volume_path_out)
            slice_num = image_in_vol.shape[3]

            print(f"Now processing for ensemble: {volume_filenames[i]}")

            # List to store predictions (probabilities) from each model in the ensemble
            ensemble_predictions_for_volume = []

            # Iterate through each model in the ensemble
            for model_args_path in self.args_json_paths:
                with open(model_args_path, 'r') as f:
                    args_dict = json.load(f)
                args = Namespace(**args_dict) # Convert dict to Namespace

                sam_model = self._load_model(args)
                print(f"Loading model from: {model_args_path}")

                # Initialize prediction array for the current model
                current_model_probabilities = np.zeros((self.num_classes, image_in_vol.shape[1], image_in_vol.shape[2], slice_num))

                for j in range(slice_num):
                    img_in_slice = image_in_vol.data[0][:, :, j]
                    img_out_slice = image_out_vol.data[0][:, :, j]

                    original_pil_img = Image.fromarray(np.stack((img_in_slice, img_out_slice, img_in_slice), axis=-1), 'RGB')

                    processed_img_tensor = torch.unsqueeze(self._preprocess_image(img_in_slice, img_out_slice, args.normalize_type), 0).to(self.device)

                    with torch.no_grad():
                        img_emb = sam_model.image_encoder(processed_img_tensor)
                        sparse_emb, dense_emb = sam_model.prompt_encoder(points=None, boxes=None, masks=None)
                        pred, _ = sam_model.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=True,
                        )

                    mask_pred_tensor = pred.cpu().float()
                    # Apply softmax to raw predictions (logits) to get probabilities
                    # Resulting shape will be (1, num_classes, H, W)
                    softmax_probabilities_slice = scipy.special.softmax(mask_pred_tensor.numpy(), axis=1)[0] # Shape (num_classes, H, W)

                    for cls_idx in range(self.num_classes):
                        # Resize probability map back to original slice dimensions
                        pil_mask = Image.fromarray(softmax_probabilities_slice[cls_idx, :, :]).resize(original_pil_img.size)
                        current_model_probabilities[cls_idx, :, :, j] = np.asarray(pil_mask)

                ensemble_predictions_for_volume.append(current_model_probabilities)

            # --- Ensemble Aggregation ---
            # stack all predictions from models: (num_models, num_classes, H, W, D)
            stacked_predictions = np.asarray(ensemble_predictions_for_volume)

            # Mean prediction (averaged probabilities)
            mean_pred_probabilities = np.mean(stacked_predictions, axis=0) # (num_classes, H, W, D)

            # Variance prediction (mean variance across classes for visualization, could be per class)
            variance_pred = np.var(stacked_predictions, axis=0) # (num_classes, H, W, D)
            # Take mean across classes to get a single variance map (H, W, D)
            mean_variance_map = np.mean(variance_pred, axis=0) # (H, W, D)

            # Resize variance map to original resolution (if not already during prediction)
            resized_mean_variance_map = np.zeros((image_in_vol.shape[1], image_in_vol.shape[2], slice_num))
            for slice_idx in range(slice_num):
                pil_mask_var = Image.fromarray(mean_variance_map[:, :, slice_idx]).resize(original_pil_img.size)
                resized_mean_variance_map[:, :, slice_idx] = np.asarray(pil_mask_var)


            # Define save paths for ensemble results
            Path(self.save_base_path).mkdir(parents=True, exist_ok=True)

            # Save mean probabilities and variance map in .npz format
            npz_filename = f"Besta_00{npz_counter + 1}.npz"
            np.savez(os.path.join(self.save_base_path, npz_filename),
                     mean_prediction=mean_pred_probabilities,
                     variance_prediction=resized_mean_variance_map)
            npz_counter += 1

            # Convert mean probabilities to a single class segmentation mask (argmax)
            # Transpose to (D, H, W) for argmax, then (1, D, H, W) for torchio
            final_mask_argmax = mean_pred_probabilities.argmax(axis=0) # (H, W, D)
            mask_tensor_for_tio = torch.tensor(final_mask_argmax, dtype=torch.int).unsqueeze(0) # (1, H, W, D)

            # Save the combined mask volume as NIfTI
            combined_mask = tio.LabelMap(tensor=mask_tensor_for_tio, affine=image_in_vol.affine)
            mask_nifti_saving_path = os.path.join(self.save_base_path, volume_filenames[i])
            combined_mask.save(mask_nifti_saving_path)

            # Load GT and original volume for plotting
            gt_path = os.path.join(self.images_gt_path, gt_filenames[i//2])
            gt_vol = load_nifti_file(gt_path)
            input_vol_for_plot = load_nifti_file(volume_path_in)

            # Create folder for example images
            images_examples_dir = os.path.join(self.save_base_path, 'Images_Examples')
            Path(images_examples_dir).mkdir(parents=True, exist_ok=True)

            plot_base_name = os.path.splitext(gt_filenames[i//2])[0]
            save_plot_path = os.path.join(images_examples_dir, plot_base_name)

            #plot the uncertainty maps

            plot_uncertainty(input_vol_for_plot, final_mask_argmax, gt_vol, mean_pred_probabilities, save_plot_path, 20)
            plot_uncertainty(input_vol_for_plot, final_mask_argmax, gt_vol, mean_pred_probabilities, save_plot_path, 40)
            plot_uncertainty(input_vol_for_plot, final_mask_argmax, gt_vol, mean_pred_probabilities, save_plot_path, 55)
