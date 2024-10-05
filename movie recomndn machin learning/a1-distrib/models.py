# models.py

from sentiment_data import SentimentExample
from utils import Indexer
from collections import Counter, defaultdict
import numpy as np
from typing import List

# Base class for all classifiers
class SentimentClassifier:
    def predict(self, sentence: List[str]) -> int:
        raise NotImplementedError("Override me")

# Perceptron Classifier
class PerceptronClassifier(SentimentClassifier):
    def __init__(self, weights=None, feature_extractor=None, indexer=None):
        self.weights = weights if weights is not None else defaultdict(float)
        self.feature_extractor = feature_extractor
        self.indexer = indexer

    def predict(self, sentence: List[str]) -> int:
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[feat] * value for feat, value in features.items())
        return 1 if score >= 0 else 0

# Logistic Regression Classifier
class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, weights, feature_extractor, indexer):
        self.weights = weights
        self.feature_extractor = feature_extractor
        self.indexer = indexer

    def predict(self, sentence: List[str]) -> int:
        features = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[feat] * value for feat, value in features.items())
        return 1 if score >= 0 else 0

# Feature Extractor Base Class
class FeatureExtractor:
    def get_indexer(self):
        raise NotImplementedError("Override me")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        raise NotImplementedError("Override me")

# Unigram Feature Extractor
class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        for word in sentence:
            if add_to_indexer:
                feature_idx = self.indexer.add_and_get_index(word)
            else:
                feature_idx = self.indexer.index_of(word)
            if feature_idx != -1:
                features[feature_idx] += 1
        return features

# Bigram Feature Extractor
class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            if add_to_indexer:
                feature_idx = self.indexer.add_and_get_index(bigram)
            else:
                feature_idx = self.indexer.index_of(bigram)
            if feature_idx != -1:
                features[feature_idx] += 1
        return features

# Combined Unigram and Bigram Feature Extractor
class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        
        # Unigram features
        for word in sentence:
            if add_to_indexer:
                feature_idx = self.indexer.add_and_get_index(word)
            else:
                feature_idx = self.indexer.index_of(word)
            if feature_idx != -1:
                features[feature_idx] += 1

        # Bigram features
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            if add_to_indexer:
                feature_idx = self.indexer.add_and_get_index(bigram)
            else:
                feature_idx = self.indexer.index_of(bigram)
            if feature_idx != -1:
                features[feature_idx] += 1
        
        return features

# Rule-Based Sentiment Classifier
class RuleBasedSentimentClassifier(SentimentClassifier):
    def __init__(self):
        pass

    def predict(self, sentence: List[str]) -> int:
        sent = " ".join(sentence)
        if "good" in sent or "great" in sent or "awesome" in sent:
            return 1
        else:
            return 0

# Training Logistic Regression Model
def train_logistic_regression_model(train_exs: List[SentimentExample], feature_extractor: FeatureExtractor) -> SentimentClassifier:
    indexer = feature_extractor.get_indexer()
    weights = defaultdict(float)
    learning_rate = 0.1
    num_epochs = 50

    for epoch in range(num_epochs):
        for ex in train_exs:
            features = feature_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[feat] * value for feat, value in features.items())
            predicted_label = 1 if score >= 0 else 0
            error = ex.label - predicted_label
            for feat, value in features.items():
                weights[feat] += learning_rate * error * value

    return LogisticRegressionClassifier(weights, feature_extractor, indexer)

# Training Perceptron Model
def train_perceptron_model(train_exs: List[SentimentExample], feature_extractor: FeatureExtractor) -> SentimentClassifier:
    indexer = feature_extractor.get_indexer()
    weights = defaultdict(float)
    learning_rate = 0.1
    num_epochs = 20

    for epoch in range(num_epochs):
        for ex in train_exs:
            features = feature_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[feat] * value for feat, value in features.items())
            predicted_label = 1 if score >= 0 else 0
            error = ex.label - predicted_label
            for feat, value in features.items():
                weights[feat] += learning_rate * error * value

    return PerceptronClassifier(weights, feature_extractor, indexer)

# Main Training Function
def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    indexer = Indexer()
    if args.feats == "UNIGRAM":
        feature_extractor = UnigramFeatureExtractor(indexer)
    elif args.feats == "BIGRAM":
        feature_extractor = BigramFeatureExtractor(indexer)
    else:
        feature_extractor = BetterFeatureExtractor(indexer)

    # Index features on the training data
    for ex in train_exs:
        feature_extractor.extract_features(ex.words, add_to_indexer=True)

    # Train the appropriate model
    if args.model == "LR":
        model = train_logistic_regression_model(train_exs, feature_extractor)
    elif args.model == "PERCEPTRON":
        model = train_perceptron_model(train_exs, feature_extractor)
    elif args.model == "TRIVIAL":
        model = RuleBasedSentimentClassifier()
    else:
        raise NotImplementedError(f"Unknown model type: {args.model}")
    
    return model

# Evaluation Function
def evaluate(dev_exs: List[SentimentExample], classifier: SentimentClassifier) -> float:
    correct = 0
    for ex in dev_exs:
        prediction = classifier.predict(ex.words)
        if prediction == ex.label:
            correct += 1
    return correct / len(dev_exs)
