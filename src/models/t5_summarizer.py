# src/models/t5_summarizer.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

class T5Summarizer:
    def __init__(self, model_name="t5-small", device=None):
        """Initialize T5 model for summarization"""
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def save_model(self, output_dir):
        """Save the model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        
    def load_model(self, model_dir):
        """Load a fine-tuned model"""
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        print(f"Model loaded from {model_dir}")
        
    def summarize(self, text, max_length=64, num_beams=4, no_repeat_ngram_size=2, 
                 early_stopping=True):
        """Generate a summary for the given text"""
        # Prepare input
        # T5 requires a "summarize: " prefix for summarization tasks
        input_text = f"summarize: {text}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping
        )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
        
    def batch_summarize(self, texts, max_length=64, batch_size=8):
        """Generate summaries for a list of texts"""
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Prepare inputs with the summarize prefix
            batch_input_texts = [f"summarize: {text}" for text in batch_texts]
            batch_inputs = self.tokenizer(batch_input_texts, padding=True, truncation=True, 
                                        return_tensors="pt").to(self.device)
            
            # Generate summaries
            batch_summary_ids = self.model.generate(
                batch_inputs['input_ids'],
                attention_mask=batch_inputs['attention_mask'],
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            # Decode summaries
            batch_summaries = self.tokenizer.batch_decode(batch_summary_ids, skip_special_tokens=True)
            summaries.extend(batch_summaries)
            
        return summaries


