# src/data/preprocess.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import yaml
import os

class DataPreprocessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Ensure NLTK resources are downloaded
        nltk.download('punkt')
        nltk.download('stopwords')
        
    def load_data(self, file_path):
        """Load dataset from file"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
    def clean_text(self, text):
        """Clean text by removing special characters, normalizing whitespace etc."""
        if pd.isna(text):
            return ""
            
        # Remove special characters
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def extract_keywords(self, text, top_n=10):
        """Extract important keywords from text using TF-IDF"""
        vectorizer = CountVectorizer(stop_words='english', 
                                   ngram_range=(1, 2), 
                                   max_features=100)
        
        # Handle single document case
        if isinstance(text, str):
            text = [text]
            
        try:
            X = vectorizer.fit_transform(text)
            words = vectorizer.get_feature_names_out()
            
            # For single document, just get the terms with highest counts
            if len(text) == 1:
                counts = X.toarray()[0]
                keywords = sorted(zip(words, counts), key=lambda x: x[1], reverse=True)[:top_n]
                return [word for word, count in keywords]
            else:
                # For multiple documents, this would need more sophisticated approach
                # like TF-IDF weighting
                return []
        except:
            return []
    
    def preprocess_dataset(self, data, abstract_col='abstract', title_col='title'):
        """Preprocess the dataset for training"""
        # Clean abstracts and titles
        data['clean_abstract'] = data[abstract_col].apply(self.clean_text)
        
        if title_col in data.columns:
            data['clean_title'] = data[title_col].apply(self.clean_text)
        
        # Extract keywords from abstracts
        data['keywords'] = data['clean_abstract'].apply(self.extract_keywords)
        
        # Filter out too short or empty abstracts
        data = data[data['clean_abstract'].str.len() > self.config['min_abstract_length']]
        
        # Prepare target summaries (using titles as target if available)
        if title_col in data.columns and self.config['use_title_as_target']:
            data['target'] = data['clean_title']
        else:
            # Use first sentence as target summary if no title
            data['target'] = data['clean_abstract'].apply(
                lambda x: sent_tokenize(x)[0] if len(sent_tokenize(x)) > 0 else x
            )
        
        return data
    
    def split_data(self, data, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
        """Split data into train, validation and test sets"""
        # First split off test set
        train_val, test = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        
        # Then split train_val into train and validation
        relative_val_size = val_size / (train_size + val_size)
        train, val = train_test_split(
            train_val, test_size=relative_val_size, random_state=random_state
        )
        
        return train, val, test
    
    def save_splits(self, train, val, test, output_dir):
        """Save data splits to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        train.to_csv(f"{output_dir}/train.csv", index=False)
        val.to_csv(f"{output_dir}/val.csv", index=False)
        test.to_csv(f"{output_dir}/test.csv", index=False)
        
        print(f"Saved {len(train)} training samples, {len(val)} validation samples, "
              f"and {len(test)} test samples to {output_dir}")
        
    def process(self, input_file, output_dir):
        """Run full preprocessing pipeline"""
        print(f"Loading data from {input_file}...")
        data = self.load_data(input_file)
        
        print(f"Preprocessing {len(data)} samples...")
        processed_data = self.preprocess_dataset(data)
        
        print("Splitting data...")
        train, val, test = self.split_data(
            processed_data, 
            train_size=self.config['train_size'],
            val_size=self.config['val_size'], 
            test_size=self.config['test_size']
        )
        
        print("Saving splits...")
        self.save_splits(train, val, test, output_dir)
        
        return train, val, test
    
def main():
    parser = argparse.ArgumentParser(description='Preprocess research paper datasets')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file path (.json)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.config)
    preprocessor.process(args.input, args.output)

if __name__ == "__main__":
    import argparse
    main()