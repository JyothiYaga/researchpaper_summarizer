# src/evaluation/metrics.py
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge import Rouge
import numpy as np
import pandas as pd

class SummarizationEvaluator:
    def __init__(self):
        """Initialize evaluator with required NLTK resources"""
        # Ensure NLTK resources are downloaded
        nltk.download('punkt')
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        
    def calculate_rouge(self, references, hypotheses):
        """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
        try:
            # Rouge requires non-empty strings
            valid_pairs = [(r, h) for r, h in zip(references, hypotheses) 
                          if len(r.strip()) > 0 and len(h.strip()) > 0]
            
            if not valid_pairs:
                return {
                    'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                }
                
            valid_refs, valid_hyps = zip(*valid_pairs)
            scores = self.rouge.get_scores(valid_hyps, valid_refs, avg=True)
            return scores
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }
    
    def calculate_bleu(self, references, hypotheses):
        """Calculate BLEU scores"""
        scores = []
        
        for ref, hyp in zip(references, hypotheses):
            try:
                # Tokenize reference and hypothesis
                ref_tokens = word_tokenize(ref)
                hyp_tokens = word_tokenize(hyp)
                
                if not hyp_tokens:
                    scores.append(0.0)
                    continue
                
                # Calculate BLEU
                score = sentence_bleu([ref_tokens], hyp_tokens, 
                                     smoothing_function=self.smooth)
                scores.append(score)
            except:
                scores.append(0.0)
                
        # Return average BLEU score
        return np.mean(scores) if scores else 0.0
    
    def calculate_keyword_preservation(self, abstracts, summaries, keywords_lists):
        """Calculate keyword preservation score"""
        preservation_scores = []
        
        for abstract, summary, keywords in zip(abstracts, summaries, keywords_lists):
            # If no keywords, skip
            if not keywords:
                preservation_scores.append(0.0)
                continue
                
            # Count keywords in summary
            summary_lower = summary.lower()
            preserved_keywords = sum(1 for kw in keywords if kw.lower() in summary_lower)
            
            # Calculate preservation score (0 to 1)
            score = preserved_keywords / len(keywords) if keywords else 0.0
            preservation_scores.append(score)
            
        # Return average preservation score
        return np.mean(preservation_scores) if preservation_scores else 0.0
    
    def evaluate(self, references, hypotheses, abstracts=None, keywords_lists=None):
        """Evaluate summaries using multiple metrics"""
        results = {}
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge(references, hypotheses)
        results['rouge-1-f'] = rouge_scores['rouge-1']['f']
        results['rouge-2-f'] = rouge_scores['rouge-2']['f']
        results['rouge-l-f'] = rouge_scores['rouge-l']['f']
        
        # Calculate BLEU score
        results['bleu'] = self.calculate_bleu(references, hypotheses)
        
        # Calculate keyword preservation if available
        if abstracts and keywords_lists:
            results['keyword_preservation'] = self.calculate_keyword_preservation(
                abstracts, hypotheses, keywords_lists
            )
            
        return results