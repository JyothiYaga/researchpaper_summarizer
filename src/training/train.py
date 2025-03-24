# src/training/train.py
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
#from transformers.optimization import AdamW 
# If the above doesn't work, try this alternative
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import os
import yaml
import time
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=64):
        """Dataset for text summarization using T5"""
        self.tokenizer = tokenizer
        self.data = data
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        abstract = self.data.iloc[idx]['clean_abstract']
        target = self.data.iloc[idx]['target']
        
        # T5 requires "summarize: " prefix for summarization tasks
        input_text = f"summarize: {abstract}"
        
        # Tokenize inputs and targets
        input_encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer.encode_plus(
            target,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten(),
            'abstract': abstract,
            'target': target
        }

class Trainer:
    def __init__(self, config_path):
        """Initialize the trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.model_name = self.config['model']['name']
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        
        # Training parameters
        self.max_input_length = self.config['model']['max_input_length']
        self.max_target_length = self.config['model']['max_target_length']
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.learning_rate = self.config['training']['learning_rate']
        self.output_dir = self.config['paths']['output_dir']
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load train and validation data"""
        train_path = self.config['paths']['train_data']
        val_path = self.config['paths']['val_data']
        
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        
        logger.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")
        
        # Create datasets
        train_dataset = SummarizationDataset(
            train_data, self.tokenizer, 
            self.max_input_length, self.max_target_length
        )
        val_dataset = SummarizationDataset(
            val_data, self.tokenizer,
            self.max_input_length, self.max_target_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate the model on validation data"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self):
        """Train the model for specified number of epochs"""
        train_loader, val_loader = self.load_data()
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            
            start_time = time.time()
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss = self.evaluate(val_loader)
            end_time = time.time()
            
            epoch_time = end_time - start_time
            logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Model checkpoint saved to {checkpoint_dir}")
                
        # Save final model
        final_dir = os.path.join(self.output_dir, "final-model")
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")
                
        return best_val_loss

# Add these lines at the beginning of your main() function
def main():
    print("Starting training script...")
    parser = argparse.ArgumentParser(description='Train T5 model for summarization')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to training configuration file')
    
    args = parser.parse_args()
    print(f"Using config file: {args.config}")
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Config file {args.config} not found!")
        return
        
    print("Initializing trainer...")
    trainer = Trainer(args.config)
    print("Starting training...")
    trainer.train()
    print("Training completed!")
