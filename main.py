from argparse import ArgumentParser
import logging
import sys
from train import train  # Import your training function from train.py

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main(args):
    setup_logging()
    # Log the start of the training process
    logging.info("Starting the training process...")

    # Call the training function from train.py
    train(args)  # Pass other hyperparameters as needed

    # Log the end of the training process
    logging.info("Training process completed.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a neural network.",
        epilog="Example usage:\n"
               "  python3 main.py --categories PER ORG LOC" ,
        conflict_handler="resolve")
    
    # Model specific arguments
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--model-name", type=str, default="prajjwal1/bert-tiny", help="Specify which model to fine-tune")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU usage.")
    
    # Data specific arguments.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--language-filter", type=str, default=None, help="When specified, all other languages are filtered from the dataset")
    group.add_argument("--categories", nargs="+", type=str, default=None, help="When specified, all other categories are filtered from the dataset. Provide a whitespace separated list of categories",)
    
    
    args = parser.parse_args()

    if args.language_filter and args.categories:
        parser.error("Both --language-filter and --categories can not be specified.")
        
    main(args)