# src/data/download.py
import argparse
import os
import json
import requests
import pandas as pd
import time
from tqdm import tqdm
import arxiv

def download_arxiv(output_path, categories=None, max_results=5000):
    """
    Download paper metadata from arXiv API
    
    Args:
        output_path: Path to save the downloaded data
        categories: List of arXiv categories to download (e.g., ['cs.AI', 'cs.CL'])
        max_results: Maximum number of papers to download
    """
    if categories is None:
        # Default to computer science and machine learning categories
        categories = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG']
    
    print(f"Downloading up to {max_results} papers from arXiv in categories: {categories}")
    
    # Create the query string
    query = ' OR '.join([f'cat:{cat}' for cat in categories])
    
    # Initialize arxiv client
    client = arxiv.Client()
    
    # Create search
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    # Download papers
    papers = []
    
    try:
        print("Fetching papers... (this may take a while)")
        for paper in tqdm(client.results(search), total=max_results):
            papers.append({
                'id': paper.entry_id.split('/')[-1],
                'title': paper.title,
                'abstract': paper.summary,
                'categories': paper.categories,
                'authors': [author.name for author in paper.authors],
                'published_date': paper.published.strftime('%Y-%m-%d'),
                'updated_date': paper.updated.strftime('%Y-%m-%d') if paper.updated else None
            })
            
            # Sleep to avoid hitting rate limits
            time.sleep(0.1)
            
            if len(papers) >= max_results:
                break
                
    except Exception as e:
        print(f"Error during download: {e}")
        # Save what we have so far
        if papers:
            print(f"Saving {len(papers)} papers collected before error occurred")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully downloaded {len(papers)} papers to {output_path}")
    
    # Also save as CSV for easier inspection
    csv_path = output_path.replace('.json', '.csv')
    pd.DataFrame(papers).to_csv(csv_path, index=False)
    print(f"Also saved as CSV to {csv_path}")
    
    return papers

def download_pubmed(output_path, query="machine learning", max_results=5000):
    """
    Download paper metadata from PubMed
    
    Args:
        output_path: Path to save the downloaded data
        query: Search query
        max_results: Maximum number of papers to download
    """
    print(f"PubMed download not yet implemented. Saving empty dataset to {output_path}")
    
    # Placeholder for PubMed implementation
    with open(output_path, 'w') as f:
        json.dump([], f)

def main():
    parser = argparse.ArgumentParser(description='Download research paper datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['arxiv', 'pubmed'],
                      help='Dataset to download (arxiv or pubmed)')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path (.json)')
    parser.add_argument('--max_results', type=int, default=5000,
                      help='Maximum number of papers to download')
    
    args = parser.parse_args()
    
    if args.dataset == 'arxiv':
        download_arxiv(args.output, max_results=args.max_results)
    elif args.dataset == 'pubmed':
        download_pubmed(args.output, max_results=args.max_results)
    else:
        print(f"Unknown dataset: {args.dataset}")

if __name__ == "__main__":
    main()