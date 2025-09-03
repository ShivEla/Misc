"""
Binary Independence Model (BIM) Information Retrieval System
============================================================

This module implements the Binary Independence Model, a probabilistic
information retrieval model that uses relevance feedback to improve
document ranking.

Features:
- Text preprocessing with stopword removal
- Term-document frequency matrix construction
- Two-stage BIM ranking (with and without relevance feedback)
- Probabilistic weight calculation
- Relevance feedback incorporation
- Performance metrics and evaluation
"""

import os
import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import time


class BinaryIndependenceModel:
    def __init__(self):
        self.documents_df = None
        self.term_doc_matrix = None
        self.vectorizer = None
        self.vocabulary = []
        self.N = 0  # Total number of documents
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
        self.N = len(self.documents_df)
        
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
        self.N = len(self.documents_df)
        
        return self.documents_df
    
    def build_term_document_matrix(self):
        """
        Builds a binary term-document matrix using CountVectorizer.
        
        Returns:
            np.array: Binary term-document matrix
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded.")
        
        processed_docs = self.documents_df['processed_text'].tolist()
        self.vectorizer = CountVectorizer(binary=True)  # Binary matrix
        term_doc_matrix = self.vectorizer.fit_transform(processed_docs)
        
        self.term_doc_matrix = term_doc_matrix.toarray()
        self.vocabulary = self.vectorizer.get_feature_names_out()
        
        return self.term_doc_matrix
    
    def calculate_initial_weights(self):
        """
        Calculates initial term weights without relevance information.
        Uses the formula: log((N - df_i) / df_i)
        
        Returns:
            np.array: Initial term weights
        """
        if self.term_doc_matrix is None:
            self.build_term_document_matrix()
        
        # Document frequency for each term
        df = np.sum(self.term_doc_matrix > 0, axis=0)
        
        # Calculate initial weights with smoothing
        weights = np.log((self.N - df + 0.5) / (df + 0.5))
        
        return weights
    
    def calculate_relevance_weights(self, relevant_docs):
        """
        Calculates term weights with relevance feedback.
        Uses the BIM formula with relevance information.
        
        Args:
            relevant_docs (list): List of relevant document indices
            
        Returns:
            np.array: Relevance-based term weights
        """
        if self.term_doc_matrix is None:
            self.build_term_document_matrix()
        
        if not relevant_docs:
            return self.calculate_initial_weights()
        
        # Convert to numpy array for indexing
        relevant_docs = np.array(relevant_docs)
        
        # Number of relevant documents
        R = len(relevant_docs)
        
        # Document frequency for each term
        df = np.sum(self.term_doc_matrix > 0, axis=0)
        
        # Frequency in relevant documents
        r = np.sum(self.term_doc_matrix[relevant_docs, :] > 0, axis=0)
        
        # BIM formula with smoothing
        # w_i = log((r_i + 0.5) / (R - r_i + 0.5)) - log((df_i - r_i + 0.5) / (N - df_i - R + r_i + 0.5))
        term1 = np.log((r + 0.5) / (R - r + 0.5))
        term2 = np.log((df - r + 0.5) / (self.N - df - R + r + 0.5))
        
        weights = term1 - term2
        
        return weights
    
    def search_stage1(self, query_text, top_k=None):
        """
        Performs stage 1 search without relevance feedback.
        
        Args:
            query_text (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            pd.DataFrame: Ranked search results
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded.")
        
        start_time = time.time()
        
        # Build term-document matrix if not already built
        if self.term_doc_matrix is None:
            self.build_term_document_matrix()
        
        # Process query
        processed_query = self.preprocess_text(query_text)
        query_vector = self.vectorizer.transform([processed_query]).toarray()
        
        # Calculate initial weights
        initial_weights = self.calculate_initial_weights()
        
        # Weight the term-document matrix and query vector
        weighted_matrix = self.term_doc_matrix * initial_weights
        weighted_query = query_vector * initial_weights
        
        # Calculate similarities (using cosine similarity)
        similarities = cosine_similarity(weighted_query, weighted_matrix)[0]
        
        search_time = time.time() - start_time
        
        # Create results DataFrame
        results_df = self.documents_df.copy()
        results_df['similarity_score'] = similarities
        results_df['search_time'] = search_time
        results_df['stage'] = 1
        
        # Sort by similarity score
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        # Return top-k results if specified
        if top_k is not None:
            results_df = results_df.head(top_k)
        
        return results_df
    
    def search_stage2(self, query_text, relevant_docs, top_k=None):
        """
        Performs stage 2 search with relevance feedback.
        
        Args:
            query_text (str): Query text
            relevant_docs (list): List of relevant document IDs
            top_k (int): Number of top results to return
            
        Returns:
            pd.DataFrame: Ranked search results with relevance feedback
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded.")
        
        start_time = time.time()
        
        # Build term-document matrix if not already built
        if self.term_doc_matrix is None:
            self.build_term_document_matrix()
        
        # Process query
        processed_query = self.preprocess_text(query_text)
        query_vector = self.vectorizer.transform([processed_query]).toarray()
        
        # Calculate relevance weights
        relevance_weights = self.calculate_relevance_weights(relevant_docs)
        
        # Weight the term-document matrix and query vector
        weighted_matrix = self.term_doc_matrix * relevance_weights
        weighted_query = query_vector * relevance_weights
        
        # Calculate similarities
        similarities = cosine_similarity(weighted_query, weighted_matrix)[0]
        
        search_time = time.time() - start_time
        
        # Create results DataFrame
        results_df = self.documents_df.copy()
        results_df['similarity_score'] = similarities
        results_df['search_time'] = search_time
        results_df['stage'] = 2
        results_df['relevant_feedback'] = [doc_id in relevant_docs for doc_id in results_df['docID']]
        
        # Sort by similarity score
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        # Return top-k results if specified
        if top_k is not None:
            results_df = results_df.head(top_k)
        
        return results_df
    
    def iterative_search(self, query_text, initial_relevant_docs=None, iterations=2, top_k_feedback=5):
        """
        Performs iterative search with automatic relevance feedback.
        
        Args:
            query_text (str): Query text
            initial_relevant_docs (list): Initial relevant documents (optional)
            iterations (int): Number of feedback iterations
            top_k_feedback (int): Number of top documents to consider as relevant in each iteration
            
        Returns:
            dict: Results from each iteration
        """
        results_history = {}
        
        # Stage 1: Initial search
        stage1_results = self.search_stage1(query_text)
        results_history['stage1'] = stage1_results
        
        # Use initial relevant docs or top results from stage 1
        if initial_relevant_docs is not None:
            current_relevant = initial_relevant_docs
        else:
            current_relevant = stage1_results.head(top_k_feedback)['docID'].tolist()
        
        # Iterative refinement
        for i in range(iterations):
            stage_name = f'stage2_iter{i+1}'
            
            # Search with current relevant documents
            stage2_results = self.search_stage2(query_text, current_relevant)
            results_history[stage_name] = stage2_results
            
            # Update relevant documents for next iteration (pseudo-relevance feedback)
            current_relevant = stage2_results.head(top_k_feedback)['docID'].tolist()
        
        return results_history
    
    def evaluate_performance(self, queries_with_relevance, use_relevance_feedback=True):
        """
        Evaluates the performance of the BIM system.
        
        Args:
            queries_with_relevance (list): List of tuples (query, relevant_doc_ids, threshold)
            use_relevance_feedback (bool): Whether to use relevance feedback
            
        Returns:
            dict: Performance metrics
        """
        if not queries_with_relevance:
            raise ValueError("No queries provided for evaluation")
        
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_map_scores = []
        
        for query_info in queries_with_relevance:
            if len(query_info) == 3:
                query, relevant_docs, threshold = query_info
            else:
                query, relevant_docs = query_info
                threshold = 0.1
            
            if use_relevance_feedback and len(relevant_docs) > 0:
                # Use some relevant docs for feedback (simulate user feedback)
                feedback_docs = relevant_docs[:max(1, len(relevant_docs)//2)]
                results = self.search_stage2(query, feedback_docs)
            else:
                results = self.search_stage1(query)
            
            # Apply threshold
            retrieved_docs = results[results['similarity_score'] >= threshold]['docID'].tolist()
            
            if not retrieved_docs and not relevant_docs:
                precision = recall = f1 = map_score = 1.0
            elif not retrieved_docs:
                precision = recall = f1 = map_score = 0.0
            elif not relevant_docs:
                precision = recall = f1 = map_score = 0.0
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
                
                # Calculate MAP (Mean Average Precision)
                map_score = self.calculate_average_precision(results, relevant_docs)
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
            all_map_scores.append(map_score)
        
        return {
            'average_precision': sum(all_precisions) / len(all_precisions),
            'average_recall': sum(all_recalls) / len(all_recalls),
            'average_f1_score': sum(all_f1_scores) / len(all_f1_scores),
            'mean_average_precision': sum(all_map_scores) / len(all_map_scores),
            'individual_results': list(zip(queries_with_relevance, all_precisions, all_recalls, all_f1_scores, all_map_scores))
        }
    
    def calculate_average_precision(self, results, relevant_docs):
        """
        Calculates Average Precision for a single query.
        
        Args:
            results (pd.DataFrame): Search results sorted by similarity
            relevant_docs (list): List of relevant document IDs
            
        Returns:
            float: Average Precision score
        """
        if not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        precision_sum = 0.0
        relevant_retrieved = 0
        
        for i, (_, row) in enumerate(results.iterrows()):
            if row['docID'] in relevant_set:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0
    
    def get_model_statistics(self):
        """
        Returns statistics about the BIM model.
        
        Returns:
            dict: Model statistics
        """
        stats = {}
        
        if self.documents_df is not None:
            stats['total_documents'] = len(self.documents_df)
        
        if self.vocabulary is not None and len(self.vocabulary) > 0:
            stats['vocabulary_size'] = len(self.vocabulary)
        
        if self.term_doc_matrix is not None:
            stats['matrix_shape'] = self.term_doc_matrix.shape
            stats['matrix_sparsity'] = (self.term_doc_matrix == 0).sum() / self.term_doc_matrix.size
            stats['average_doc_length'] = np.mean(np.sum(self.term_doc_matrix, axis=1))
            stats['average_term_frequency'] = np.mean(np.sum(self.term_doc_matrix, axis=0))
        
        return stats
    
    def compare_stages(self, query_text, relevant_docs, top_k=10):
        """
        Compares results from Stage 1 and Stage 2 for a given query.
        
        Args:
            query_text (str): Query text
            relevant_docs (list): List of relevant document IDs
            top_k (int): Number of top results to compare
            
        Returns:
            dict: Comparison results
        """
        # Get results from both stages
        stage1_results = self.search_stage1(query_text, top_k)
        stage2_results = self.search_stage2(query_text, relevant_docs, top_k)
        
        # Calculate metrics for both stages
        stage1_retrieved = set(stage1_results['docID'].tolist())
        stage2_retrieved = set(stage2_results['docID'].tolist())
        relevant_set = set(relevant_docs)
        
        # Stage 1 metrics
        stage1_precision = len(stage1_retrieved & relevant_set) / len(stage1_retrieved) if stage1_retrieved else 0
        stage1_recall = len(stage1_retrieved & relevant_set) / len(relevant_set) if relevant_set else 0
        
        # Stage 2 metrics
        stage2_precision = len(stage2_retrieved & relevant_set) / len(stage2_retrieved) if stage2_retrieved else 0
        stage2_recall = len(stage2_retrieved & relevant_set) / len(relevant_set) if relevant_set else 0
        
        return {
            'stage1_results': stage1_results,
            'stage2_results': stage2_results,
            'stage1_precision': stage1_precision,
            'stage1_recall': stage1_recall,
            'stage2_precision': stage2_precision,
            'stage2_recall': stage2_recall,
            'improvement_precision': stage2_precision - stage1_precision,
            'improvement_recall': stage2_recall - stage1_recall
        }


def create_sample_documents():
    """
    Creates sample documents for testing the BIM system.
    
    Returns:
        pd.DataFrame: Sample documents
    """
    sample_docs = [
        "Information requirement query considers user feedback for better retrieval.",
        "Information retrieval query depends on retrieval model and user interaction.",
        "Prediction problem many problems in retrieval as prediction and machine learning.",
        "Search engine one application of retrieval models and information systems.",
        "Feedback improves query prediction and enhances user experience significantly.",
        "Machine learning algorithms help improve information retrieval effectiveness.",
        "User feedback is crucial for relevance assessment in information systems.",
        "Query expansion techniques use relevance feedback to improve search results.",
        "Probabilistic models in information retrieval use statistical methods.",
        "Binary independence model assumes terms are independent given relevance."
    ]
    
    return pd.DataFrame({
        'docID': range(len(sample_docs)),
        'text': sample_docs
    })


def demo_binary_independence_model():
    """
    Demonstrates the Binary Independence Model with sample data.
    """
    print("Binary Independence Model (BIM) Demo")
    print("=" * 50)
    
    # Initialize the system
    bim = BinaryIndependenceModel()
    
    # Load sample documents
    print("Loading sample documents...")
    sample_df = create_sample_documents()
    bim.load_documents_from_dataframe(sample_df)
    
    # Build term-document matrix
    print("Building term-document matrix...")
    matrix = bim.build_term_document_matrix()
    
    # Display model statistics
    stats = bim.get_model_statistics()
    print(f"\nModel Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example query
    query = "feedback improves query prediction"
    print(f"\nDemo Query: '{query}'")
    print("=" * 40)
    
    # Stage 1: Initial ranking
    print("\nStage 1: Initial Ranking (no relevance feedback)")
    stage1_results = bim.search_stage1(query, top_k=5)
    print("Top 5 Results:")
    for _, row in stage1_results.iterrows():
        print(f"  Doc {row['docID']}: {row['similarity_score']:.4f}")
    
    # Simulate relevance feedback (assume top 2 docs are relevant)
    relevant_docs = stage1_results.head(2)['docID'].tolist()
    print(f"\nAssuming documents {relevant_docs} are relevant...")
    
    # Stage 2: Ranking with relevance feedback
    print("\nStage 2: Ranking with Relevance Feedback")
    stage2_results = bim.search_stage2(query, relevant_docs, top_k=5)
    print("Top 5 Results:")
    for _, row in stage2_results.iterrows():
        print(f"  Doc {row['docID']}: {row['similarity_score']:.4f}")
    
    # Compare stages
    print(f"\nStage Comparison:")
    print("-" * 20)
    comparison = bim.compare_stages(query, relevant_docs, top_k=5)
    print(f"Stage 1 Precision: {comparison['stage1_precision']:.3f}")
    print(f"Stage 2 Precision: {comparison['stage2_precision']:.3f}")
    print(f"Precision Improvement: {comparison['improvement_precision']:.3f}")
    
    # Iterative search demo
    print(f"\nIterative Search Demo:")
    print("-" * 25)
    iterative_results = bim.iterative_search(query, iterations=2, top_k_feedback=3)
    
    for stage, results in iterative_results.items():
        print(f"\n{stage}: Top 3 documents")
        top_3 = results.head(3)
        for _, row in top_3.iterrows():
            print(f"  Doc {row['docID']}: {row['similarity_score']:.4f}")
    
    # Performance evaluation
    print(f"\nPerformance Evaluation:")
    print("-" * 30)
    
    # Define test queries with relevance judgments
    test_queries = [
        ("feedback query prediction", [0, 4, 7], 0.1),
        ("information retrieval model", [1, 3, 5], 0.1),
        ("machine learning algorithms", [2, 5], 0.1),
    ]
    
    # Evaluate without relevance feedback
    performance_stage1 = bim.evaluate_performance(test_queries, use_relevance_feedback=False)
    print("Stage 1 Performance (no feedback):")
    print(f"  Average Precision: {performance_stage1['average_precision']:.3f}")
    print(f"  Average Recall: {performance_stage1['average_recall']:.3f}")
    print(f"  Average F1-Score: {performance_stage1['average_f1_score']:.3f}")
    print(f"  Mean Average Precision: {performance_stage1['mean_average_precision']:.3f}")
    
    # Evaluate with relevance feedback
    performance_stage2 = bim.evaluate_performance(test_queries, use_relevance_feedback=True)
    print("\nStage 2 Performance (with feedback):")
    print(f"  Average Precision: {performance_stage2['average_precision']:.3f}")
    print(f"  Average Recall: {performance_stage2['average_recall']:.3f}")
    print(f"  Average F1-Score: {performance_stage2['average_f1_score']:.3f}")
    print(f"  Mean Average Precision: {performance_stage2['mean_average_precision']:.3f}")


if __name__ == "__main__":
    demo_binary_independence_model()
