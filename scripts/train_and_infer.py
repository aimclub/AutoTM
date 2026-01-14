import pandas as pd
import logging
from autotm.base import AutoTM
import argparse
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train and Infer AutoTM Topic Model")
    parser.add_argument("--dataset_path", type=str, default="data/sample_corpora/sample_dataset_lenta.csv", help="Path to the dataset CSV file")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column containing text")
    parser.add_argument("--lang", type=str, default="ru", help="Language of the dataset (ru or en)")
    parser.add_argument("--topic_count", type=int, default=10, help="Number of topics")
    parser.add_argument("--model_path", type=str, default="autotm_model", help="Path to save/load the model")
    parser.add_argument("--mode", type=str, choices=["train", "infer", "train_infer"], default="train_infer", help="Mode: train, infer, or train_infer")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for GA")

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset not found at {args.dataset_path}")
        return

    df = pd.read_csv(args.dataset_path)
    logger.info(f"Loaded dataset with {len(df)} rows")

    working_dir = "autotm_workdir"
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    if args.mode in ["train", "train_infer"]:
        logger.info("Starting training...")
        autotm = AutoTM(
            topic_count=args.topic_count,
            preprocessing_params={
                "lang": args.lang
            },
            texts_column_name=args.text_column,
            alg_name="ga", # Using Genetic Algorithm as default
            alg_params={"num_iterations": args.iterations},
            working_dir_path=working_dir,
            exp_dataset_name="custom_dataset"
        )
        
        # Fit and Predict
        mixtures = autotm.fit_predict(df)
        logger.info("Training complete.")
        logger.info(f"Top 5 mixtures:\n{mixtures.head().to_string()}")
        
        autotm.save(args.model_path, overwrite=True)
        logger.info(f"Model saved to {args.model_path}")
        
        print("\nExtracted Topics:")
        autotm.print_topics()

    if args.mode in ["infer", "train_infer"]:
        if not os.path.exists(args.model_path) and args.mode == "infer":
            logger.error(f"Model path {args.model_path} does not exist. Train first.")
            return

        logger.info("Starting inference with loaded model...")
        # Load model
        autotm_loaded = AutoTM.load(args.model_path)
        
        # Predict
        mixtures = autotm_loaded.predict(df)
        logger.info("Inference complete.")
        logger.info(f"Top 5 mixtures:\n{mixtures.head().to_string()}")

if __name__ == "__main__":
    main()


