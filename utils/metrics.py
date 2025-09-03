
import nibabel as nib
from medpy import metric
import matplotlib.pylab as plt
from tensorflow.keras.utils import to_categorical
from utils import reliability_diagram
from skimage.measure import regionprops
from scipy.ndimage import label
import torchio as tio
import numpy as np
import pandas as pd
import os
from PIL import Image

NUM_CLASSES = 13

row_labels = [
    "Background", "Vastus Lateralis", "Vastus Medialis", "Vastus Intermedius",
    "Rectus Femoris", "Sartorius", "Gracilis", "Adductor Magnus",
    "Semimembranosus", "Semitendinosus", "Biceps Femoris Long",
    "Biceps Femoris Short", "Adductor Longus", "Mean DSC"
]
classes = row_labels.copy()


# ------------- Helper Functions -------------------

def generalized_dice_coeff(y_true, y_pred, eps=np.finfo(float).eps, reduce_along_batch=False,
                           reduce_along_features=True, feature_weights=None, threshold=None, keepdims=False):
    y_true = y_true[:, :, :, 1:]
    y_pred = y_pred[:, :, :, 1:]
    ndim = y_true.ndim
    ax = tuple(range(1, ndim - 1))

    if threshold is not None:
        y_true = (y_true > threshold).astype(y_true.dtype)
        y_pred = (y_pred > threshold).astype(y_pred.dtype)

    intersection = np.sum(y_true * y_pred, axis=ax, keepdims=keepdims)
    denom = np.sum(y_true, axis=ax, keepdims=keepdims) + np.sum(y_pred, axis=ax, keepdims=keepdims)

    if reduce_along_features:
        if feature_weights is None:
            feature_weights = 1

        intersection = np.sum(intersection * feature_weights, axis=-1, keepdims=keepdims)
        denom = np.sum(denom * feature_weights, axis=-1, keepdims=keepdims)

    if reduce_along_batch:
        intersection = np.sum(intersection, axis=0, keepdims=keepdims)
        denom = np.sum(denom, axis=0, keepdims=keepdims)

    return (2 * intersection + eps) / (denom + eps)


def dice_coef_single_label(class_idx: int, name: str):
    def dice_coef(y_true: np.ndarray, y_pred: np.ndarray):
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)

        y_true_single_class = y_true[..., class_idx]
        y_pred_single_class = y_pred[..., class_idx]

        return metric.binary.dc(y_pred_single_class, y_true_single_class)

    dice_coef.__name__ = f"dice_coef_{name}"
    return dice_coef


def load_mask(mask_path):
    mask_img = nib.load(mask_path)
    affine = mask_img.affine
    header = mask_img.header
    mask = np.array(mask_img.get_fdata()).astype('int32')

    cat_mask = np.eye(NUM_CLASSES)[mask]
    cat_mask = np.transpose(cat_mask, (2, 0, 1, 3))  # [slices, H, W, classes]

    return cat_mask, header, affine


# ------------- Main Class -------------------

class ValMetrics:
    def __init__(self, gt_path, args_path=None, severity_case='default', save_path=None, fold_name=None):
        self.labels_folder = gt_path
        self.severity_case = severity_case
        self.predicted_folder = save_path
        self.fold_name = fold_name
        self.args_path = args_path  # Can be used for config if needed

    def evaluate(self):
        test_folders_lab = sorted([f for f in os.listdir(self.labels_folder) if f.endswith('.nii.gz')])

        test_folders_pred = sorted([f for f in os.listdir(self.predicted_folder) if f.endswith('.nii.gz')])
        df_list = []

        for i, test_patient in enumerate(test_folders_lab):
            print(f'Processing file_{i + 1}: {test_patient}')

            path_gt = os.path.join(self.labels_folder, test_patient)
            path_pred = os.path.join(self.predicted_folder, test_folders_pred[i])

            ground_truth, info_mask, _ = load_mask(path_gt)
            mask_predicted, _, _ = load_mask(path_pred)

            labeled_indexes = np.any((1 - ground_truth[:, :, :, 0]), axis=(1, 2))
            true_indices = np.where(labeled_indexes == True)[0]
            # evaluation only on the labeled slices
            gt_clean = ground_truth[true_indices]
            pred_clean = mask_predicted[true_indices]

            dice_multi = []
            hd = []
            assd = []

            for idx, class_name in enumerate(classes):
                if idx == len(classes) - 1:  # Mean Dice
                    dice_value = generalized_dice_coeff(
                        pred_clean, gt_clean,
                        reduce_along_batch=True, reduce_along_features=True
                    )
                    hd_value = np.nan
                    assd_value = np.nan
                else:
                    dice_fn = dice_coef_single_label(class_idx=idx, name=class_name)
                    dice_value = dice_fn(gt_clean, pred_clean)

                    if np.unique(pred_clean[:, :, :, idx]).shape[0] == 1 or \
                       np.unique(gt_clean[:, :, :, idx]).shape[0] == 1:
                        hd_value = np.nan
                        assd_value = np.nan
                    else:
                        hd_value = float(metric.hd95(pred_clean[:, :, :, idx], gt_clean[:, :, :, idx],
                                                     voxelspacing=info_mask.get_zooms()))
                        assd_value = float(metric.assd(pred_clean[:, :, :, idx], gt_clean[:, :, :, idx],
                                                       voxelspacing=info_mask.get_zooms()))

                dice_multi.append(dice_value)
                hd.append(hd_value)
                assd.append(assd_value)

            df = pd.DataFrame({
                'Class Name': row_labels,
                'Dice Value': dice_multi,
                'Hausdorff Distance': hd,
                'ASSD': assd
            })

            empty_df = pd.DataFrame([[" "] * df.shape[1]], columns=df.columns)
            df_list.append(df)
            df_list.append(empty_df)

        final_df = pd.concat(df_list)
        output_path = os.path.join(self.predicted_folder, "metrics.csv")
        final_df.to_csv(output_path, index=False)
        print(f"Saved metrics to: {output_path}")
        print(f"Saved metrics to: {output_path}")

    def evaluate_calibration(self):

        names = sorted([f for f in os.listdir(self.labels_folder) if f.endswith('.nii.gz')])
        names_pred = [f.replace(".nii.gz", ".npz") for f in names]  # match prediction files
        num_classes = 13
        num_bins = 20

        NLL_tot, ECE_tot, Brier_tot, Entropy = [], [], [], []
        x_dataset, y_dataset = [], []
        y_dataset_pred = np.empty((0, num_classes))

        save_path = os.path.join(self.predicted_folder, "uncertainty")
        os.makedirs(save_path, exist_ok=True)

        for idx, gt_file in enumerate(names):
            gt_img = tio.ScalarImage(os.path.join(self.labels_folder, gt_file)).data[0].numpy()
            gt_img = np.transpose(gt_img, (2, 0, 1))  # slice, h, w

            pred = np.load(os.path.join(self.predicted_folder, names_pred[idx]))
            pred_mean = np.transpose(pred['mean_prediction'], (3, 1, 2, 0))  # slice, h, w, classes
            labels_pred = np.argmax(pred_mean, axis=-1)

            for kk in range(labels_pred.shape[0]):
                ytrue, ypred = [], []
                conf = np.empty((0, num_classes))

                labels_true_one = np.where(gt_img[kk, ...] > 0, 1, 0)
                labels_true_mod, lab_num = label(labels_true_one) #binarizza e poi genera componenti connesse, crea bb attorno alle componenti

                if lab_num > 0:
                    bb = regionprops(labels_true_mod.astype('int32'))
                    for region in bb:
                        mina_cl, minb_cl, maxa_cl, maxb_cl = region.bbox
                        mina_cl = max(0, mina_cl - 5)
                        minb_cl = max(0, minb_cl - 5)
                        maxa_cl = min(gt_img.shape[1], maxa_cl + 5)
                        maxb_cl = min(gt_img.shape[2], maxb_cl + 5)

                        ytrue_crop = gt_img[kk, mina_cl:maxa_cl, minb_cl:maxb_cl]
                        ypred_crop = labels_pred[kk, mina_cl:maxa_cl, minb_cl:maxb_cl]
                        conf_crop = pred_mean[kk, mina_cl:maxa_cl, minb_cl:maxb_cl, :]

                        ytrue = np.append(ytrue, ytrue_crop.ravel())
                        ypred = np.append(ypred, ypred_crop.ravel())
                        conf = np.append(conf, conf_crop.reshape(-1, num_classes), axis=0)

                x_dataset = np.append(x_dataset, ytrue)
                y_dataset = np.append(y_dataset, ypred)
                y_dataset_pred = np.append(y_dataset_pred, conf, axis=0)

        ytrue = x_dataset.astype('int32')
        ypred = y_dataset.astype('int32')
        conf = y_dataset_pred.astype('float32')
        ytrue_OHE = to_categorical(ytrue, num_classes=num_classes)
        conf = np.clip(conf, 0, 1)
        conf_max = np.max(conf, axis=1)

        Brier_multiclass = np.mean(np.sum((conf - ytrue_OHE) ** 2, axis=1))
        bin_data = reliability_diagram.compute_calibration(ytrue, ypred, conf_max, num_bins)
        ECE_multiclass = bin_data["expected_calibration_error"]
        NLL_multiclass = reliability_diagram.NLL(ytrue_OHE, conf)
        epsilon = 1e-10
        mean_entropy = np.mean(-np.sum(conf * np.log(conf + epsilon), axis=1))

        ECE_tot.append(ECE_multiclass)
        NLL_tot.append(NLL_multiclass)
        Brier_tot.append(Brier_multiclass)
        Entropy.append(mean_entropy)

        pd.DataFrame({
            "ECE": ECE_tot,
            "NLL": NLL_tot,
            "Brier": Brier_tot,
            "Entropy": Entropy
        }).to_csv(os.path.join(save_path, "calibration_score.csv"))

        fig = reliability_diagram.reliability_diagram(ytrue, ypred, conf_max, num_bins=20,
                                                      draw_ece=True, draw_bin_importance="alpha",
                                                      draw_averages=True, title='Calibration',
                                                      figsize=(6, 6), dpi=100, return_fig=True)
        fig.savefig(os.path.join(save_path, 'overall_calibration_plot.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Per-class metrics
        ECE_cls = np.empty(num_classes - 1)
        NLL_cls = np.empty(num_classes - 1)
        for i in range(1, num_classes):
            ytrue1 = (ytrue == i).astype('int32')
            ypred1 = (ypred == i).astype('int32')
            conf1 = conf[:, i]
            confnll = np.stack([1 - conf1, conf1], axis=1)

            bin_data = reliability_diagram.compute_calibration(ytrue1, ypred1, conf1, num_bins)
            ECE_cls[i - 1] = bin_data["expected_calibration_error"]
            ytrue_OHE_bin = to_categorical(ytrue1, num_classes=2)
            NLL_cls[i - 1] = reliability_diagram.NLL(ytrue_OHE_bin, confnll)

            fig = reliability_diagram.reliability_diagram(ytrue1, ypred1, conf1, num_bins=20,
                                                          draw_ece=True, draw_bin_importance="alpha",
                                                          draw_averages=True, title=row_labels[i - 1],
                                                          figsize=(6, 6), dpi=100, return_fig=True)
            fig.savefig(os.path.join(save_path, f'calibration_plot_class_{i}.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)

        pd.DataFrame({
            "ECE": ECE_cls,
            "NLL": NLL_cls
        }, index=[row_labels[1:-1]]).to_csv(os.path.join(save_path, "calibration_score_class.csv"))