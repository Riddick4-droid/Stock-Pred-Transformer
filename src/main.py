"""
Command‑Line Interface for the Stock Transformer Project
Usage:
    python main.py ingest  # Download & prepare data
    python main.py features  # Run feature engineering pipeline
    python main.py pretrain  # Pre‑train on all stocks
    python main.py finetune  # Fine‑tune on a single stock
    python main.py evaluate  # Evaluate on test set
    python main.py predict   # Run single & multi‑step inference
"""

import argparse
import sys
from pathlib import Path

# Make sure the project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_ingestion import run_data_digestion
from src.feature_engineering.features import run_feature_pipeline
from src.train_model.trainer import PreTrainer, FineTuner
from src.model_eval.evaluate import evaluate
from src.inference import run_inference
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Stock Transformer – Production‑grade CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    #  ingest 
    subparsers.add_parser("ingest", help="Download & cache raw data (Kaggle + VIX)")

    #  features 
    subparsers.add_parser("features", help="Run feature engineering on all raw stocks")

    # pretrain
    pretrain_parser = subparsers.add_parser("pretrain", help="Pre‑train the transformer on all stocks")
    pretrain_parser.add_argument("--config", default="configs/config.yaml",
                                 help="Path to config file")

    #finetune 
    finetune_parser = subparsers.add_parser("finetune", help="Fine‑tune on a single stock")
    finetune_parser.add_argument("--config", default="configs/config.yaml",
                                 help="Path to config file")

    #  evaluate 
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--config", default="configs/config.yaml",
                             help="Path to config file")
    eval_parser.add_argument("--mode", choices=["pre_training", "fine_tuning"],
                             default="fine_tuning", help="Which phase to evaluate")
    eval_parser.add_argument("--checkpoint", default=None,
                             help="Path to checkpoint (default: from config)")
    eval_parser.add_argument("--output", default="results",
                             help="Directory for metrics & plots")

    #predict
    predict_parser = subparsers.add_parser("predict", help="Run inference on the latest data")
    predict_parser.add_argument("--config", default="configs/config.yaml",
                                help="Path to config file")
    predict_parser.add_argument("--ticker", default=None,
                                help="Ticker to predict (default: from config)")
    predict_parser.add_argument("--checkpoint", default=None,
                                help="Path to checkpoint (default: from config)")

    args = parser.parse_args()

    
    #route to the correct function
    if args.command == "ingest":
        logger.info("Running data ingestion...")
        run_data_digestion()
        logger.info("Ingestion complete.")

    elif args.command == "features":
        logger.info("Running feature pipeline...")
        run_feature_pipeline()
        logger.info("Feature pipeline complete.")

    elif args.command == "pretrain":
        logger.info("Starting pretraining...")
        trainer = PreTrainer(config_path=args.config)
        trainer.train()
        logger.info("Pre‑training finished.")

    elif args.command == "finetune":
        logger.info("Starting finetuning...")
        finetuner = FineTuner(config_path=args.config)
        finetuner.run()
        logger.info("Fine‑tuning finished.")

    elif args.command == "evaluate":
        logger.info("Running evaluation...")
        metrics = evaluate(
            config_path=args.config,
            mode=args.mode,
            checkpoint_path=args.checkpoint,
            output_dir=args.output
        )
        logger.info(f"Metrics: {metrics}")

    elif args.command == "predict":
        logger.info("Running inference...")
        run_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            ticker=args.ticker
        )
        logger.info("Inference complete.")


if __name__ == "__main__":
    main()