import os
import json
from pathlib import Path
import torch

from config.test_settings import parse_args
from inference.single_fold_inference import SAMSingleFoldTester
from inference.ensemble_inference import SAMEnsembleTester
from utils.metrics import ValMetrics

def main():

    # Ensure checkpoint directory exists and save args

    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", "MedSAM", "Decoder") #choose the correct path where the folds are

    args = parse_args()

    # --- Single Fold Testing ---
    print("\n--- Starting Single Fold Testing ---")
    list_folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

    for fold_name in list_folds:
        print(f"\nProcessing {fold_name}...")

        # Construct the args.json path for the current fold
        # This assumes each fold's checkpoint directory has its own args.json
        # E.g., ./checkpoints/Lora/fold_1/args.json
        current_fold_args_path = os.path.join(checkpoint_dir, fold_name, "args.json")

        # Healthy
        print(f"Testing Healthy case for {fold_name}")
        tester_healthy = SAMSingleFoldTester(
            images_ts_path=args.test_image_healthy_path,
            images_gt_path=args.test_gt_healthy_path,
            severity_case='healthy',
            args_json_path=current_fold_args_path,
            save_base_path=os.path.join(checkpoint_dir, fold_name, 'predictions', 'healthy'),
            fold_name=fold_name
        )

        tester_healthy.predict_volume()


        val_metrics_healthy = ValMetrics(
            gt_path=args.test_gt_healthy_path,
            args_path=current_fold_args_path,
            severity_case='healthy',
            save_path=os.path.join(checkpoint_dir, fold_name, 'predictions', 'healthy'),
            fold_name=fold_name
        )
        val_metrics_healthy.evaluate()
        val_metrics_healthy.evaluate_calibration()

        # Moderate
        print(f"Testing Moderate case for {fold_name}")
        tester_moderate = SAMSingleFoldTester(
            images_ts_path=args.test_image_moderate_path,
            images_gt_path=args.test_gt_moderate_path,
            severity_case='moderate',
            args_json_path=current_fold_args_path,
            save_base_path=os.path.join(checkpoint_dir, fold_name, 'predictions', 'moderate'),
            fold_name=fold_name
        )

        tester_moderate.predict_volume()

        val_metrics_moderate = ValMetrics(
            gt_path=args.test_gt_moderate_path,
            args_path=current_fold_args_path,
            severity_case='moderate',
            save_path=os.path.join(checkpoint_dir, fold_name, 'predictions', 'moderate'),
            fold_name=fold_name
        )
        val_metrics_moderate.evaluate()
        val_metrics_moderate.evaluate_calibration()

        # Severe
        print(f"Testing Severe case for {fold_name}")
        tester_severe = SAMSingleFoldTester(
            images_ts_path=args.test_image_severe_path,
            images_gt_path=args.test_gt_severe_path,
            severity_case='severe',
            args_json_path=current_fold_args_path,
            save_base_path=os.path.join(checkpoint_dir, fold_name, 'predictions', 'severe'),
            fold_name=fold_name
        )

        tester_severe.predict_volume()

        val_metrics_severe = ValMetrics(
            gt_path=args.test_gt_severe_path,
            args_path=current_fold_args_path,
            severity_case='severe',
            save_path=os.path.join(checkpoint_dir, fold_name, 'predictions', 'severe'),
            fold_name=fold_name
        )

        val_metrics_severe.evaluate()
        val_metrics_severe.evaluate_calibration()
    #

    # --- Ensemble Testing ---
    print("\n--- Starting Ensemble Testing ---")
    # Paths to args.json for each fold for the ensemble
    ensemble_args_paths = [os.path.join(checkpoint_dir, f_name, "args.json") for f_name in list_folds]

    # Healthy Ensemble
    print("Testing Healthy case for Ensemble")

    ensemble_tester_healthy = SAMEnsembleTester(
        images_ts_path=args.test_image_healthy_path,
        images_gt_path=args.test_gt_healthy_path,
        severity_case='healthy',
        args_json_paths=ensemble_args_paths,
        save_base_path=os.path.join(checkpoint_dir, 'ensemble', 'predictions', 'healthy')
    )
    ensemble_tester_healthy.predict_ensemble()

    # For ensemble, metrics evaluation might use the first args_path or a dedicated ensemble args
    val_metrics_ensemble_healthy = ValMetrics(
        gt_path=args.test_gt_healthy_path,
        args_path=ensemble_args_paths[0], # Using first args for general settings (e.g., num_cls)
        severity_case='healthy',
        save_path=os.path.join(checkpoint_dir, 'ensemble', 'predictions', 'healthy'),
        fold_name='ensemble' # Special fold name for ensemble results
    )
    val_metrics_ensemble_healthy.evaluate()
    val_metrics_ensemble_healthy.evaluate_calibration()

    # Moderate Ensemble
    print("Testing Moderate case for Ensemble")
    ensemble_tester_moderate = SAMEnsembleTester(
        images_ts_path=args.test_image_moderate_path,
        images_gt_path=args.test_gt_moderate_path,
        severity_case='moderate',
        args_json_paths=ensemble_args_paths,
        save_base_path=os.path.join(checkpoint_dir, 'ensemble', 'predictions', 'moderate')
    )

    ensemble_tester_moderate.predict_ensemble()

    val_metrics_ensemble_moderate = ValMetrics(
        gt_path=args.test_gt_moderate_path,
        args_path=ensemble_args_paths[0],
        severity_case='moderate',
        save_path=os.path.join(checkpoint_dir, 'ensemble', 'predictions', 'moderate'),
        fold_name='ensemble'
    )
    val_metrics_ensemble_moderate.evaluate()
    val_metrics_ensemble_moderate.evaluate_calibration()
    # Severe Ensemble
    print("Testing Severe case for Ensemble")
    ensemble_tester_severe = SAMEnsembleTester(
        images_ts_path=args.test_image_severe_path,
        images_gt_path=args.test_gt_severe_path,
        severity_case='severe',
        args_json_paths=ensemble_args_paths,
        save_base_path=os.path.join(checkpoint_dir, 'ensemble', 'predictions', 'severe')
    )

    ensemble_tester_severe.predict_ensemble()

    val_metrics_ensemble_severe = ValMetrics(
        gt_path=args.test_gt_severe_path,
        args_path=ensemble_args_paths[0],
        severity_case='severe',
        save_path=os.path.join(checkpoint_dir, 'ensemble', 'predictions', 'severe'),
        fold_name='ensemble'
    )
    val_metrics_ensemble_severe.evaluate()
    val_metrics_ensemble_severe.evaluate_calibration()

if __name__ == "__main__":
    main()