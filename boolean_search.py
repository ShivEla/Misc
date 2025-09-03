"""
Boolean Search Information Retrieval System
============================================

This module implements a Boolean search model using inverted indexing
for document retrieval based on AND, OR, and NOT operations.

Features:
- Text preprocessing with stopword removal
- Inverted index construction
- Boolean query processing (AND, OR, NOT)
- Performance metrics (precision, recall, F1-score)
- Document ranking and retrieval
"""

import os
import re
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import time


class BooleanSearchSystem:
    def __init__(self):
        self.inverted_index = {}
        self.documents_df = None
        self.stop_words = ['the', 'a', 'is', 'in', 'of', 'and', 'to', 'it', 'for', 'that', 
                          'with', 'on', 'as', 'are', 'was', 'be', 'been', 'have', 'has',
                          'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should']
    
    def preprocess_text(self, text):
        """
        Preprocesses text by converting to lowercase, removing punctuation,
        and filtering out stop words.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]
        return ' '.join(filtered_tokens)
    
    def load_documents_from_directory(self, directory_path):
        """
        Loads documents from a directory of text files.
        
        Args:
            directory_path (str): Path to directory containing text files
            
        Returns:
            pd.DataFrame: DataFrame with document IDs and text content
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        file_list = sorted([f for f in os.listdir(directory_path) if f.endswith('.txt')])
        
        if not file_list:
            raise ValueError("No .txt files found in the directory")
        
        texts = []
        filenames = []
        
        for filename in file_list:
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    texts.append(content)
                    filenames.append(filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        
        self.documents_df = pd.DataFrame({
            'docID': range(len(texts)),
            'filename': filenames,
            'text': texts
        })
        
        # Add processed text column
        self.documents_df['processed_text'] = self.documents_df['text'].apply(self.preprocess_text)
        
        return self.documents_df
    
    def load_documents_from_dataframe(self, df, text_column='text', id_column=None):
        """
        Loads documents from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing documents
            text_column (str): Name of column containing text content
            id_column (str): Name of column containing document IDs (optional)
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        self.documents_df = df.copy()
        
        if id_column is None:
            self.documents_df['docID'] = range(len(df))
        else:
            self.documents_df['docID'] = df[id_column]
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        self.documents_df['text'] = df[text_column]
        self.documents_df['processed_text'] = self.documents_df['text'].apply(self.preprocess_text)
        
        return self.documents_df
    
    def build_inverted_index(self):
        """
        Builds an inverted index from the loaded documents.
        
        Returns:
            dict: Inverted index mapping terms to document IDs
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded. Use load_documents_from_directory() or load_documents_from_dataframe() first.")
        
        inverted_index = defaultdict(list)
        
        for index, row in self.documents_df.iterrows():
            doc_id = row['docID']
            terms = row['processed_text'].split()
            
            # Use set to avoid duplicate document IDs for the same term
            for term in set(terms):
                inverted_index[term].append(doc_id)
        
        # Sort document IDs for each term to enable efficient intersection
        for term in inverted_index:
            inverted_index[term].sort()
        
        self.inverted_index = dict(inverted_index)
        return self.inverted_index
    
    def intersect(self, p1, p2):
        """
        Finds the intersection of two sorted postings lists (AND operation).
        
        Args:
            p1 (list): First postings list
            p2 (list): Second postings list
            
        Returns:
            list: Documents present in both lists
        """
        answer = []
        i, j = 0, 0
        
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                answer.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1
        
        return answer
    
    def union(self, p1, p2):
        """
        Finds the union of two sorted postings lists (OR operation).
        
        Args:
            p1 (list): First postings list
            p2 (list): Second postings list
            
        Returns:
            list: Documents from either list, with no duplicates
        """
        return sorted(list(set(p1) | set(p2)))
    
    def process_query(self, query):
        """
        Processes a Boolean query and returns matching document IDs.
        
        Args:
            query (str): Boolean query (e.g., "term1 AND term2", "term1 OR term2")
            
        Returns:
            list: List of matching document IDs
        """
        if not self.inverted_index:
            raise ValueError("Inverted index not built. Use build_inverted_index() first.")
        
        query_parts = query.lower().split()
        
        if "and" in query_parts:
            and_idx = query_parts.index("and")
            term1 = query_parts[and_idx - 1]
            term2 = query_parts[and_idx + 1]
            
            p1 = self.inverted_index.get(term1, [])
            p2 = self.inverted_index.get(term2, [])
            
            return self.intersect(p1, p2)
        
        elif "or" in query_parts:
            or_idx = query_parts.index("or")
            term1 = query_parts[or_idx - 1]
            term2 = query_parts[or_idx + 1]
            
            p1 = self.inverted_index.get(term1, [])
            p2 = self.inverted_index.get(term2, [])
            
            return self.union(p1, p2)
        
        elif "not" in query_parts:
            # Handle "term1 AND NOT term2" format
            and_idx = query_parts.index("and") if "and" in query_parts else -1
            not_idx = query_parts.index("not")
            
            if and_idx != -1 and and_idx < not_idx:
                term1 = query_parts[and_idx - 1]
                term2 = query_parts[not_idx + 1]
                
                p1 = self.inverted_index.get(term1, [])
                p2 = self.inverted_index.get(term2, [])
                
                # Find documents in the first list that are NOT in the second
                return [doc_id for doc_id in p1 if doc_id not in p2]
            else:
                # Simple NOT query
                term = query_parts[not_idx + 1]
                p1 = self.inverted_index.get(term, [])
                all_docs = set(self.documents_df['docID'].tolist())
                return sorted(list(all_docs - set(p1)))
        
        else:
            # Simple single-term query
            term = self.preprocess_text(query).split()[0] if self.preprocess_text(query) else query.lower()
            return self.inverted_index.get(term, [])
    
    def search(self, query, return_content=True):
        """
        Searches for documents matching the Boolean query.
        
        Args:
            query (str): Boolean query
            return_content (bool): Whether to return document content
            
        Returns:
            pd.DataFrame: Search results with document information
        """
        start_time = time.time()
        matching_docs = self.process_query(query)
        search_time = time.time() - start_time
        
        if not matching_docs:
            return pd.DataFrame(columns=['docID', 'filename', 'text', 'relevance_score'])
        
        results_df = self.documents_df[self.documents_df['docID'].isin(matching_docs)].copy()
        results_df['relevance_score'] = 1.0  # Boolean model: either relevant (1) or not (0)
        results_df['search_time'] = search_time
        
        if not return_content:
            results_df = results_df.drop('text', axis=1)
        
        return results_df.sort_values('docID')
    
    def evaluate_performance(self, queries_with_relevance):
        """
        Evaluates the performance of the Boolean search system.
        
        Args:
            queries_with_relevance (list): List of tuples (query, relevant_doc_ids)
            
        Returns:
            dict: Performance metrics (precision, recall, F1-score)
        """
        if not queries_with_relevance:
            raise ValueError("No queries provided for evaluation")
        
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        
        for query, relevant_docs in queries_with_relevance:
            retrieved_docs = self.process_query(query)
            
            if not retrieved_docs and not relevant_docs:
                # Perfect match for empty results
                precision = recall = f1 = 1.0
            elif not retrieved_docs:
                # No documents retrieved
                precision = recall = f1 = 0.0
            elif not relevant_docs:
                # No relevant documents (should not happen in practice)
                precision = recall = f1 = 0.0
            else:
                # Calculate metrics
                retrieved_set = set(retrieved_docs)
                relevant_set = set(relevant_docs)
                
                true_positives = len(retrieved_set & relevant_set)
                false_positives = len(retrieved_set - relevant_set)
                false_negatives = len(relevant_set - retrieved_set)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
        
        return {
            'average_precision': sum(all_precisions) / len(all_precisions),
            'average_recall': sum(all_recalls) / len(all_recalls),
            'average_f1_score': sum(all_f1_scores) / len(all_f1_scores),
            'individual_results': list(zip(queries_with_relevance, all_precisions, all_recalls, all_f1_scores))
        }
    
    def get_index_statistics(self):
        """
        Returns statistics about the inverted index.
        
        Returns:
            dict: Index statistics
        """
        if not self.inverted_index:
            return {"error": "Index not built"}
        
        vocab_size = len(self.inverted_index)
        total_postings = sum(len(postings) for postings in self.inverted_index.values())
        avg_postings_per_term = total_postings / vocab_size if vocab_size > 0 else 0
        
        return {
            'vocabulary_size': vocab_size,
            'total_postings': total_postings,
            'average_postings_per_term': avg_postings_per_term,
            'total_documents': len(self.documents_df) if self.documents_df is not None else 0
        }


def create_sample_documents():
    """
    Creates sample documents for testing the Boolean search system.
    
    Returns:
        pd.DataFrame: Sample documents
    """
    sample_docs = [
        "This is the first document about information retrieval.",
        "The second document discusses vector space models and information systems.",
        "Information retrieval models include probabilistic models and Boolean models.",
        "Search engines use various retrieval algorithms for document ranking.",
        "Query processing is an important aspect of information retrieval systems."
    ]
    
    return pd.DataFrame({
        'docID': range(len(sample_docs)),
        'text': sample_docs
    })


def demo_boolean_search():
    """
    Demonstrates the Boolean search system with sample data.
    """
    print("Boolean Search System Demo")
    print("=" * 50)
    
    # Initialize the system
    search_system = BooleanSearchSystem()
    
    # Load sample documents
    print("Loading sample documents...")
    sample_df = create_sample_documents()
    search_system.load_documents_from_dataframe(sample_df)
    
    # Build inverted index
    print("Building inverted index...")
    index = search_system.build_inverted_index()
    
    # Display index statistics
    stats = search_system.get_index_statistics()
    print(f"\nIndex Statistics:")
    print(f"Vocabulary size: {stats['vocabulary_size']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Average postings per term: {stats['average_postings_per_term']:.2f}")
    
    # Example queries
    queries = [
        "information AND retrieval",
        "vector OR models",
        "information AND NOT probabilistic",
        "document",
        "systems"
    ]
    
    print(f"\nTesting Boolean queries:")
    print("-" * 30)
    
    for query in queries:
        results = search_system.search(query, return_content=False)
        print(f"Query: '{query}'")
        print(f"Results: {len(results)} documents")
        if len(results) > 0:
            print(f"Document IDs: {results['docID'].tolist()}")
        print()
    
    # Performance evaluation with sample relevance judgments
    print("Performance Evaluation:")
    print("-" * 30)
    
    # Define relevance judgments (query, list of relevant doc IDs)
    relevance_judgments = [
        ("information AND retrieval", [0, 2, 4]),
        ("vector OR models", [1, 2]),
        ("document", [0, 1, 2, 3]),
    ]
    
    performance = search_system.evaluate_performance(relevance_judgments)
    print(f"Average Precision: {performance['average_precision']:.3f}")
    print(f"Average Recall: {performance['average_recall']:.3f}")
    print(f"Average F1-Score: {performance['average_f1_score']:.3f}")


if __name__ == "__main__":
    demo_boolean_search()
