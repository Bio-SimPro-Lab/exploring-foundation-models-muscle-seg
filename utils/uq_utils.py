# uq_utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import reliability_diagram

def evaluate_calibration(ytrue, ypred, conf, save_path, row_labels, num_bins=20):
    os.makedirs(save_path, exist_ok=True)
    num_classes = conf.shape[1]

    ytrue_OHE = to_categorical(ytrue, num_classes=num_classes)
    conf = np.clip(conf, 0, 1)

    Brier_multiclass = np.mean(np.sum((conf - ytrue_OHE) ** 2, axis=1))
    conf_max = np.max(conf, axis=1)

    bin_data = reliability_diagram.compute_calibration(ytrue, ypred, conf_max, num_bins)
    ECE_multiclass = bin_data["expected_calibration_error"]
    NLL_multiclass = reliability_diagram.NLL(ytrue_OHE, conf)

    epsilon = 1e-10
    entropy = -np.sum(conf * np.log(conf + epsilon), axis=1)
    mean_entropy = np.mean(entropy)

    # Save overall metrics
    net_metrics = {
        "ECE": [ECE_multiclass],
        "NLL": [NLL_multiclass],
        "Brier": [Brier_multiclass],
        "Entropy": [mean_entropy]
    }
    pd.DataFrame(net_metrics).to_csv(os.path.join(save_path, "calibration_score.csv"))

    fig = reliability_diagram.reliability_diagram(ytrue, ypred, conf_max, num_bins=20, draw_ece=True,
                                                  draw_bin_importance="alpha", draw_averages=True,
                                                  title='Calibration', figsize=(6, 6), dpi=100, return_fig=True)
    fig.savefig(os.path.join(save_path, 'reliability_diagram.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Class-wise diagrams
    ECE = np.empty(num_classes - 1)
    NLL = np.empty(num_classes - 1)
    for i in range(1, num_classes):
        ytrue_bin = (ytrue == i).astype(int)
        ypred_bin = (ypred == i).astype(int)
        conf_bin = conf[:, i]

        confnll = np.stack([1 - conf_bin, conf_bin], axis=1)
        ytrue_OHE_bin = to_categorical(ytrue_bin, num_classes=2)

        bin_data = reliability_diagram.compute_calibration(ytrue_bin, ypred_bin, conf_bin, num_bins)
        ECE[i - 1] = bin_data["expected_calibration_error"]
        NLL[i - 1] = reliability_diagram.NLL(ytrue_OHE_bin, confnll)

        fig = reliability_diagram.reliability_diagram(ytrue_bin, ypred_bin, conf_bin, num_bins=20, draw_ece=True,
                                                      draw_bin_importance="alpha", draw_averages=True,
                                                      title=row_labels[i], figsize=(6, 6), dpi=100,
                                                      return_fig=True)
        fig.savefig(os.path.join(save_path, f'calibration_class_{i}.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

    df_class = pd.DataFrame({
        "ECE": ECE,
        "NLL": NLL
    }, index=row_labels[1:])
    df_class.to_csv(os.path.join(save_path, "calibration_score_class.csv"))
