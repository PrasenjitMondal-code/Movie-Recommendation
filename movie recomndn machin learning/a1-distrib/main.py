import argparse
from sentiment_data import read_sentiment_examples
from models import train_model, evaluate

def main():
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Train and evaluate a sentiment analysis model.")
        parser.add_argument("--train_file", type=str, required=True, help="Path to the training data")
        parser.add_argument("--dev_file", type=str, required=True, help="Path to the development data")
        parser.add_argument("--model", type=str, choices=["LR", "PERCEPTRON", "TRIVIAL"], default="LR", help="Type of model to train")
        parser.add_argument("--feats", type=str, choices=["UNIGRAM", "BIGRAM", "BETTER"], default="UNIGRAM", help="Type of features to use")
        args = parser.parse_args()

        # Debug: Print arguments
        print(f"Arguments: {args}")

        # Read the data
        print(f"Reading training data from: {args.train_file}")
        try:
            train_exs = read_sentiment_examples(args.train_file)
            print(f"Number of training examples: {len(train_exs)}")
        except Exception as e:
            print(f"Error reading training data: {e}")
            return

        print(f"Reading development data from: {args.dev_file}")
        try:
            dev_exs = read_sentiment_examples(args.dev_file)
            print(f"Number of development examples: {len(dev_exs)}")
        except Exception as e:
            print(f"Error reading development data: {e}")
            return

        # Train the model
        print(f"Training model with type: {args.model} and features: {args.feats}")
        try:
            classifier = train_model(args, train_exs, dev_exs)
        except Exception as e:
            print(f"Error training model: {e}")
            return

        # Evaluate the model
        print("Evaluating model...")
        try:
            accuracy = evaluate(dev_exs, classifier)
            print(f"Development set accuracy: {accuracy * 100:.2f}%")
        except Exception as e:
            print(f"Error evaluating model: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
