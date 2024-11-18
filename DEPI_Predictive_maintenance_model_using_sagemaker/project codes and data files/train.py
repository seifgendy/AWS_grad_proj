import argparse
import os
import xgboost as xgb
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args):
    # Load the datasets from the SageMaker input directories
    train_path = os.path.join(args.train, 'train.csv')
    val_path = os.path.join(args.validation, 'validation.csv') if args.validation else None
    
    # Load data into DMatrix format for XGBoost
    logger.info("Loading training data...")
    dtrain = xgb.DMatrix(train_path)
    dval = xgb.DMatrix(val_path) if val_path else None
    
    # Define the watchlist
    watchlist = [(dtrain, 'train')]
    if dval:
        watchlist.append((dval, 'validation'))
    
    # Set XGBoost hyperparameters
    hyperparams = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'objective': args.objective,
        'num_class': args.num_class if args.num_class else None,
    }
    
    # Train the model
    logger.info("Starting training...")
    bst = xgb.train(params=hyperparams,
                    dtrain=dtrain,
                    num_boost_round=args.num_round,
                    evals=watchlist)
    
    # Save the model to the specified output path
    model_output = os.path.join(args.model_dir, 'xgboost-model')
    logger.info(f"Saving model to {model_output}...")
    bst.save_model(model_output)

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--objective', type=str, default='multi:softmax')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--num_round', type=int, default=100)
    
    # SageMaker specific arguments
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    args = parser.parse_args()
    
    # Run the training process
    train(args)
