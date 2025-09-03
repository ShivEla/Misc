"""
Vector Space Model (VSM) Information Retrieval System
====================================================

This module implements a Vector Space Model for document retrieval using
TF-IDF weighting and various similarity measures.

Features:
- Text preprocessing with stopword removal
- TF (Term Frequency) calculation
- IDF (Inverse Document Frequency) calculation
- TF-IDF matrix construction
- Multiple similarity measures (Cosine, Jaccard, Dice)
- Query-based document ranking
- Performance metrics and evaluation
"""

import os
import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.metrics import precision_score, recall_score, f1_score
import time


class VectorSpaceModel:
    def __init__(self):
        self.documents_df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.custom_tfidf_matrix = None
        self.tf_scores = {}
        self.idf_scores = {}
        self.vocabulary = []
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
    
    def calculate_tf(self):
        """
        Calculates Term Frequency for each term in each document.
        
        Returns:
            dict: Dictionary of TF scores for each document
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded.")
        
        tf_scores = {}
        for _, row in self.documents_df.iterrows():
            doc_id = row['docID']
            terms = row['processed_text'].split()
            term_counts = defaultdict(int)
            
            for term in terms:
                term_counts[term] += 1
            
            tf_scores[doc_id] = dict(term_counts)
        
        self.tf_scores = tf_scores
        return tf_scores
    
    def calculate_idf(self):
        """
        Calculates Inverse Document Frequency for each term in the collection.
        
        Returns:
            dict: Dictionary of IDF scores for each term
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded.")
        
        N = len(self.documents_df)
        idf_scores = {}
        all_terms = set()
        
        # Collect all unique terms
        for text in self.documents_df['processed_text']:
            all_terms.update(text.split())
        
        # Calculate IDF for each term
        for term in all_terms:
            doc_count = sum(1 for text in self.documents_df['processed_text'] 
                           if term in text.split())
            idf_scores[term] = math.log(N / doc_count) if doc_count > 0 else 0
        
        self.idf_scores = idf_scores
        return idf_scores
    
    def build_tfidf_matrix_sklearn(self):
        """
        Builds TF-IDF matrix using sklearn's TfidfVectorizer.
        
        Returns:
            pd.DataFrame: TF-IDF matrix
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded.")
        
        processed_docs = self.documents_df['processed_text'].tolist()
        self.vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm='l2')
        tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        
        terms = self.vectorizer.get_feature_names_out()
        doc_ids = self.documents_df['docID'].tolist()
        
        self.tfidf_matrix = pd.DataFrame(
            tfidf_matrix.toarray(), 
            index=doc_ids, 
            columns=terms
        )
        self.vocabulary = list(terms)
        
        return self.tfidf_matrix
    
    def build_custom_tfidf_matrix(self):
        """
        Builds TF-IDF matrix using custom TF and IDF calculations.
        
        Returns:
            pd.DataFrame: Custom TF-IDF matrix
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded.")
        
        # Calculate TF and IDF scores
        self.calculate_tf()
        self.calculate_idf()
        
        # Get all unique terms to form vocabulary
        all_terms = sorted(list(set(term for text in self.documents_df['processed_text'] 
                                   for term in text.split())))
        vocab_map = {term: i for i, term in enumerate(all_terms)}
        
        # Initialize TF-IDF matrix
        num_docs = len(self.documents_df)
        num_terms = len(all_terms)
        tfidf_matrix = np.zeros((num_docs, num_terms))
        
        # Populate matrix with TF-IDF scores
        for i, doc_id in enumerate(self.documents_df['docID']):
            doc_tf_scores = self.tf_scores.get(doc_id, {})
            for term, tf_value in doc_tf_scores.items():
                j = vocab_map[term]
                idf_value = self.idf_scores.get(term, 0)
                tfidf_matrix[i, j] = tf_value * idf_value
        
        self.custom_tfidf_matrix = pd.DataFrame(
            tfidf_matrix, 
            index=self.documents_df['docID'], 
            columns=all_terms
        )
        self.vocabulary = all_terms
        
        return self.custom_tfidf_matrix
    
    def cosine_similarity_manual(self, vec1, vec2):
        """
        Calculates cosine similarity between two vectors manually.
        
        Args:
            vec1 (np.array): First vector
            vec2 (np.array): Second vector
            
        Returns:
            float: Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    def jaccard_similarity(self, vec1, vec2):
        """
        Calculates Jaccard similarity between two binary vectors.
        
        Args:
            vec1 (np.array): First vector
            vec2 (np.array): Second vector
            
        Returns:
            float: Jaccard similarity score
        """
        # Convert to binary vectors
        bin_vec1 = (vec1 > 0).astype(int)
        bin_vec2 = (vec2 > 0).astype(int)
        
        intersection = np.sum(bin_vec1 & bin_vec2)
        union = np.sum(bin_vec1 | bin_vec2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def dice_similarity(self, vec1, vec2):
        """
        Calculates Dice similarity between two binary vectors.
        
        Args:
            vec1 (np.array): First vector
            vec2 (np.array): Second vector
            
        Returns:
            float: Dice similarity score
        """
        # Convert to binary vectors
        bin_vec1 = (vec1 > 0).astype(int)
        bin_vec2 = (vec2 > 0).astype(int)
        
        intersection = np.sum(bin_vec1 & bin_vec2)
        total_elements = np.sum(bin_vec1) + np.sum(bin_vec2)
        
        if total_elements == 0:
            return 0.0
        
        return (2 * intersection) / total_elements
    
    def search(self, query_text, similarity_measure='cosine', use_sklearn=True, top_k=None):
        """
        Searches for documents similar to the query using VSM.
        
        Args:
            query_text (str): Query text
            similarity_measure (str): Similarity measure ('cosine', 'jaccard', 'dice')
            use_sklearn (bool): Whether to use sklearn's TF-IDF or custom implementation
            top_k (int): Number of top results to return (None for all)
            
        Returns:
            pd.DataFrame: Ranked search results
        """
        if self.documents_df is None:
            raise ValueError("No documents loaded.")
        
        start_time = time.time()
        
        # Choose TF-IDF matrix
        if use_sklearn:
            if self.tfidf_matrix is None:
                self.build_tfidf_matrix_sklearn()
            tfidf_matrix = self.tfidf_matrix
            vectorizer = self.vectorizer
        else:
            if self.custom_tfidf_matrix is None:
                self.build_custom_tfidf_matrix()
            tfidf_matrix = self.custom_tfidf_matrix
            vectorizer = None
        
        # Process query
        processed_query = self.preprocess_text(query_text)
        
        if use_sklearn and vectorizer:
            # Transform query using the fitted vectorizer
            query_vector = vectorizer.transform([processed_query]).toarray()[0]
        else:
            # Manual query vector creation for custom TF-IDF
            query_terms = processed_query.split()
            query_vector = np.zeros(len(self.vocabulary))
            
            for term in query_terms:
                if term in self.vocabulary:
                    term_idx = self.vocabulary.index(term)
                    # Use TF-IDF weighting for query
                    tf_query = query_terms.count(term)
                    idf_query = self.idf_scores.get(term, 0)
                    query_vector[term_idx] = tf_query * idf_query
        
        # Calculate similarities
        similarities = []
        doc_vectors = tfidf_matrix.values
        
        for i, doc_vector in enumerate(doc_vectors):
            if similarity_measure == 'cosine':
                if use_sklearn:
                    # Use sklearn's cosine similarity
                    sim = cosine_similarity([query_vector], [doc_vector])[0][0]
                else:
                    sim = self.cosine_similarity_manual(query_vector, doc_vector)
            elif similarity_measure == 'jaccard':
                sim = self.jaccard_similarity(query_vector, doc_vector)
            elif similarity_measure == 'dice':
                sim = self.dice_similarity(query_vector, doc_vector)
            else:
                raise ValueError(f"Unknown similarity measure: {similarity_measure}")
            
            similarities.append(sim)
        
        search_time = time.time() - start_time
        
        # Create results DataFrame
        results_df = self.documents_df.copy()
        results_df['similarity_score'] = similarities
        results_df['search_time'] = search_time
        results_df['similarity_measure'] = similarity_measure
        
        # Sort by similarity score
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        # Return top-k results if specified
        if top_k is not None:
            results_df = results_df.head(top_k)
        
        return results_df
    
    def evaluate_performance(self, queries_with_relevance, similarity_measure='cosine', use_sklearn=True):
        """
        Evaluates the performance of the VSM system.
        
        Args:
            queries_with_relevance (list): List of tuples (query, relevant_doc_ids, threshold)
            similarity_measure (str): Similarity measure to use
            use_sklearn (bool): Whether to use sklearn's implementation
            
        Returns:
            dict: Performance metrics
        """
        if not queries_with_relevance:
            raise ValueError("No queries provided for evaluation")
        
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_ndcg_scores = []
        
        for query_info in queries_with_relevance:
            if len(query_info) == 3:
                query, relevant_docs, threshold = query_info
            else:
                query, relevant_docs = query_info
                threshold = 0.1  # Default threshold
            
            # Get search results
            results = self.search(query, similarity_measure, use_sklearn)
            
            # Apply threshold to determine retrieved documents
            retrieved_docs = results[results['similarity_score'] >= threshold]['docID'].tolist()
            
            if not retrieved_docs and not relevant_docs:
                precision = recall = f1 = ndcg = 1.0
            elif not retrieved_docs:
                precision = recall = f1 = ndcg = 0.0
            elif not relevant_docs:
                precision = recall = f1 = ndcg = 0.0
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
                
                # Calculate NDCG (simplified version)
                ndcg = self.calculate_ndcg(results, relevant_docs)
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
            all_ndcg_scores.append(ndcg)
        
        return {
            'average_precision': sum(all_precisions) / len(all_precisions),
            'average_recall': sum(all_recalls) / len(all_recalls),
            'average_f1_score': sum(all_f1_scores) / len(all_f1_scores),
            'average_ndcg': sum(all_ndcg_scores) / len(all_ndcg_scores),
            'individual_results': list(zip(queries_with_relevance, all_precisions, all_recalls, all_f1_scores, all_ndcg_scores))
        }
    
    def calculate_ndcg(self, results, relevant_docs, k=10):
        """
        Calculates Normalized Discounted Cumulative Gain (NDCG).
        
        Args:
            results (pd.DataFrame): Search results sorted by similarity
            relevant_docs (list): List of relevant document IDs
            k (int): Number of top results to consider
            
        Returns:
            float: NDCG score
        """
        # Get top-k results
        top_results = results.head(k)
        
        # Calculate DCG
        dcg = 0.0
        for i, (_, row) in enumerate(top_results.iterrows()):
            if row['docID'] in relevant_docs:
                relevance = 1  # Binary relevance
                dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) is 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_docs), k)):
            idcg += 1 / math.log2(i + 2)
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def get_model_statistics(self):
        """
        Returns statistics about the VSM model.
        
        Returns:
            dict: Model statistics
        """
        stats = {}
        
        if self.documents_df is not None:
            stats['total_documents'] = len(self.documents_df)
        
        if self.vocabulary:
            stats['vocabulary_size'] = len(self.vocabulary)
        
        if self.tfidf_matrix is not None:
            stats['sklearn_matrix_shape'] = self.tfidf_matrix.shape
            stats['sklearn_matrix_sparsity'] = (self.tfidf_matrix == 0).sum().sum() / self.tfidf_matrix.size
        
        if self.custom_tfidf_matrix is not None:
            stats['custom_matrix_shape'] = self.custom_tfidf_matrix.shape
            stats['custom_matrix_sparsity'] = (self.custom_tfidf_matrix == 0).sum().sum() / self.custom_tfidf_matrix.size
        
        return stats


def create_sample_documents():
    """
    Creates sample documents for testing the VSM system.
    
    Returns:
        pd.DataFrame: Sample documents
    """
    sample_docs = [
        "Information retrieval is the activity of obtaining information system resources.",
        "Vector space model represents text documents as vectors of identifiers.",
        "Term frequency inverse document frequency is a numerical statistic.",
        "Cosine similarity measures the cosine of the angle between two vectors.",
        "Natural language processing deals with interactions between computers and human language.",
        "Machine learning algorithms can improve information retrieval systems.",
        "Text mining and information extraction are related to information retrieval."
    ]
    
    return pd.DataFrame({
        'docID': range(len(sample_docs)),
        'text': sample_docs
    })


def demo_vector_space_model():
    """
    Demonstrates the Vector Space Model with sample data.
    """
    print("Vector Space Model (VSM) Demo")
    print("=" * 50)
    
    # Initialize the system
    vsm = VectorSpaceModel()
    
    # Load sample documents
    print("Loading sample documents...")
    sample_df = create_sample_documents()
    vsm.load_documents_from_dataframe(sample_df)
    
    # Build TF-IDF matrices
    print("Building TF-IDF matrices...")
    sklearn_matrix = vsm.build_tfidf_matrix_sklearn()
    custom_matrix = vsm.build_custom_tfidf_matrix()
    
    # Display model statistics
    stats = vsm.get_model_statistics()
    print(f"\nModel Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example queries with different similarity measures
    queries = [
        "information retrieval systems",
        "vector space model",
        "machine learning text",
        "cosine similarity"
    ]
    
    similarity_measures = ['cosine', 'jaccard', 'dice']
    
    print(f"\nTesting queries with different similarity measures:")
    print("-" * 60)
    
    for query in queries[:2]:  # Test first 2 queries to keep output manageable
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        for sim_measure in similarity_measures:
            results = vsm.search(query, similarity_measure=sim_measure, use_sklearn=True, top_k=3)
            print(f"\n{sim_measure.capitalize()} Similarity (Top 3):")
            for _, row in results.iterrows():
                print(f"  Doc {row['docID']}: {row['similarity_score']:.4f}")
    
    # Performance evaluation
    print(f"\nPerformance Evaluation:")
    print("-" * 30)
    
    # Define relevance judgments (query, list of relevant doc IDs, threshold)
    relevance_judgments = [
        ("information retrieval", [0, 5, 6], 0.1),
        ("vector space model", [1], 0.1),
        ("machine learning", [4, 5], 0.1),
    ]
    
    performance = vsm.evaluate_performance(relevance_judgments, similarity_measure='cosine')
    print(f"Average Precision: {performance['average_precision']:.3f}")
    print(f"Average Recall: {performance['average_recall']:.3f}")
    print(f"Average F1-Score: {performance['average_f1_score']:.3f}")
    print(f"Average NDCG: {performance['average_ndcg']:.3f}")


if __name__ == "__main__":
    demo_vector_space_model()
