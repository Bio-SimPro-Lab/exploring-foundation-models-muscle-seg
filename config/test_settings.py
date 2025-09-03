# config/test_settings.py
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="SAM Inference Testing Configuration")

    # path where the nifti files are. They are organized following the nnUnet format (e.g. for 2 channel images, DatasetName_00x_0000.nii for channel 0
    # of subject x, DatasetName_00x_0001.nii for channel 1 of the same subject.)
    # For labels, the name is just DatasetName_00x.nii
    parser.add_argument('--test_image_healthy_path', type=str,
                        default=r"C:\Users\39340\Documents\Besta_segm\nnUnetFrame\dataset\nnUnet_raw\Dataset005_Besta\Test\Healthy",
                        help='Path to healthy test images.')
    parser.add_argument('--test_gt_healthy_path', type=str,
                        default=r"C:\Users\39340\Documents\Besta_segm\nnUnetFrame\dataset\nnUnet_raw\Dataset005_Besta\Test\Healthy_labels",
                        help='Path to healthy ground truth labels.')
    parser.add_argument('--test_image_moderate_path', type=str,
                        default=r"C:\Users\39340\Documents\Besta_segm\nnUnetFrame\dataset\nnUnet_raw\Dataset005_Besta\Test\Moderate",
                        help='Path to moderate test images.')
    parser.add_argument('--test_gt_moderate_path', type=str,
                        default=r"C:\Users\39340\Documents\Besta_segm\nnUnetFrame\dataset\nnUnet_raw\Dataset005_Besta\Test\Moderate_labels",
                        help='Path to moderate ground truth labels.')
    parser.add_argument('--test_image_severe_path', type=str,
                        default=r"C:\Users\39340\Documents\Besta_segm\nnUnetFrame\dataset\nnUnet_raw\Dataset005_Besta\Test\Severe",
                        help='Path to severe test images.')
    parser.add_argument('--test_gt_severe_path', type=str,
                        default=r"C:\Users\39340\Documents\Besta_segm\nnUnetFrame\dataset\nnUnet_raw\Dataset005_Besta\Test\Severe_labels",
                        help='Path to severe ground truth labels.')



    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Whether to use GPU for inference')
    parser.add_argument('--if_vis', type=bool, default=True,
                        help='Whether to visualize plots')

    args, unknown = parser.parse_known_args()
    print(args)
    return args