"""
cross_space_alignment.py - Embedding space alignment techniques for transfer learning

This module implements methods for aligning independently trained embedding spaces while 
preserving their internal structural relationships. It provides:

1. OrthogonalProcrustes: Implements the classic Orthogonal Procrustes algorithm for 
   finding the optimal rotation matrix between two embedding spaces. This preserves 
   relative distances between vectors, maintaining the structural relationships within 
   each space. The method uses SVD (Singular Value Decomposition) to find the optimal 
   transformation that minimizes the Frobenius norm.

2. ManhattanAlignment: An alternative alignment technique based on Manhattan (L1) distance 
   optimization rather than Euclidean distance. This approach can be more robust in 
   high-dimensional spaces where the "curse of dimensionality" affects Euclidean metrics.
   Uses gradient descent with an orthogonality regularization penalty.

These techniques are useful for transfer learning scenarios where you need to map concepts 
between domains that were trained independently (e.g., specialized domain models and 
general knowledge models). The methods enable knowledge transfer while preserving the 
semantic relationships within each domain.

Note: While developed for aligning Buddhist and contemporary knowledge representations,
these techniques are applicable to any cross-domain alignment problem in NLP or 
representation learning.
"""

import tensorflow as tf
import numpy as np

class OrthogonalProcrustes:
    """
    Cross-space alignment using Orthogonal Procrustes algorithm.
    This technique aligns embedding spaces while preserving internal structures,
    enabling meaningful connections between Buddhist concepts and contemporary knowledge.
    """
    def __init__(self):
        pass
    
    def align(self, source_vectors, target_vectors):
        """
        Align source embedding space to target embedding space
        using orthogonal Procrustes algorithm.
        
        Parameters:
        - source_vectors: Matrix of shape [n_concepts, embedding_dim] for Buddhist concepts
        - target_vectors: Matrix of shape [n_concepts, embedding_dim] for contemporary concepts
        
        Returns:
        - W: Orthogonal transformation matrix
        - aligned_vectors: Source vectors aligned to target space
        """
        # Convert inputs to TensorFlow tensors if they're not already
        source_vectors = tf.convert_to_tensor(source_vectors, dtype=tf.float32)
        target_vectors = tf.convert_to_tensor(target_vectors, dtype=tf.float32)
        
        # Compute YX^T
        correlation = tf.matmul(target_vectors, source_vectors, transpose_b=True)
        
        # Compute SVD of YX^T
        s, u, v = tf.linalg.svd(correlation)
        
        # Compute the orthogonal transformation matrix W = UV^T
        W = tf.matmul(u, v, transpose_b=True)
        
        # Apply transformation to source vectors
        aligned_vectors = tf.matmul(source_vectors, W, transpose_b=True)
        
        # Calculate alignment quality (cosine similarities)
        source_norms = tf.norm(aligned_vectors, axis=1, keepdims=True)
        target_norms = tf.norm(target_vectors, axis=1, keepdims=True)
        cosine_similarities = tf.reduce_sum(
            aligned_vectors * target_vectors, axis=1
        ) / (tf.squeeze(source_norms) * tf.squeeze(target_norms))
        
        avg_similarity = tf.reduce_mean(cosine_similarities)
        
        return W, aligned_vectors, avg_similarity.numpy()


class ManhattanAlignment:
    """
    A different approach to cross-space alignment that uses Manhattan distance
    instead of Euclidean distance, which may be more robust in high dimensions.
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def align(self, source_vectors, target_vectors):
        """
        Align source embedding space to target embedding space
        by minimizing Manhattan distance through gradient descent.
        
        Parameters:
        - source_vectors: Matrix of shape [n_concepts, embedding_dim] for Buddhist concepts
        - target_vectors: Matrix of shape [n_concepts, embedding_dim] for contemporary concepts
        
        Returns:
        - W: Transformation matrix
        - aligned_vectors: Source vectors aligned to target space
        """
        # Convert inputs to TensorFlow tensors if they're not already
        source_vectors = tf.convert_to_tensor(source_vectors, dtype=tf.float32)
        target_vectors = tf.convert_to_tensor(target_vectors, dtype=tf.float32)
        
        # Get dimensions
        n_concepts, embedding_dim = source_vectors.shape
        
        # Initialize the transformation matrix
        W = tf.Variable(
            tf.random.normal((embedding_dim, embedding_dim), stddev=0.1)
        )
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                # Apply transformation
                aligned_vectors = tf.matmul(source_vectors, W)
                
                # Compute Manhattan distance
                manhattan_distance = tf.reduce_mean(
                    tf.reduce_sum(tf.abs(aligned_vectors - target_vectors), axis=1)
                )
                
                # Add regularization for orthogonality (optional)
                orthogonality_penalty = tf.reduce_mean(
                    tf.abs(tf.matmul(W, W, transpose_b=True) - tf.eye(embedding_dim))
                )
                
                # Total loss
                loss = manhattan_distance + 0.1 * orthogonality_penalty
            
            # Compute gradients and update W
            gradients = tape.gradient(loss, [W])
            optimizer.apply_gradients(zip(gradients, [W]))
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.numpy()}")
        
        # Apply final transformation
        aligned_vectors = tf.matmul(source_vectors, W)
        
        # Calculate alignment quality (cosine similarities)
        source_norms = tf.norm(aligned_vectors, axis=1, keepdims=True)
        target_norms = tf.norm(target_vectors, axis=1, keepdims=True)
        cosine_similarities = tf.reduce_sum(
            aligned_vectors * target_vectors, axis=1
        ) / (tf.squeeze(source_norms) * tf.squeeze(target_norms))
        
        avg_similarity = tf.reduce_mean(cosine_similarities)
        
        return W, aligned_vectors, avg_similarity.numpy()


def demonstrate_cross_space_alignment():
    """
    Example usage of cross-space alignment techniques
    """
    # For demonstration purposes - in practice these would be large embedding matrices
    # Sample Buddhist concept vectors (simplified for demonstration)
    buddhist_vectors = np.array([
        [0.2, 0.5, 0.1, 0.8],  # emptiness (śūnyatā)
        [0.3, 0.2, 0.7, 0.1],  # dependent origination (pratītyasamutpāda)
        [0.9, 0.1, 0.3, 0.2]   # mindfulness (smṛti)
    ], dtype=np.float32)
    
    # Sample contemporary concept vectors
    contemporary_vectors = np.array([
        [0.25, 0.45, 0.2, 0.7],  # non-essentialism
        [0.35, 0.25, 0.6, 0.2],  # systems theory
        [0.8, 0.15, 0.4, 0.3]    # present-moment awareness
    ], dtype=np.float32)
    
    # Orthogonal Procrustes alignment
    procrustes = OrthogonalProcrustes()
    W_procrustes, aligned_vectors_procrustes, avg_sim_procrustes = procrustes.align(
        buddhist_vectors, contemporary_vectors
    )
    
    print(f"Orthogonal Procrustes alignment average similarity: {avg_sim_procrustes:.4f}")
    
    # Manhattan alignment
    manhattan = ManhattanAlignment(epochs=500)
    W_manhattan, aligned_vectors_manhattan, avg_sim_manhattan = manhattan.align(
        buddhist_vectors, contemporary_vectors
    )
    
    print(f"Manhattan alignment average similarity: {avg_sim_manhattan:.4f}")
    
    # Demonstrate transformation of a new Buddhist concept
    new_buddhist_concept = np.array([[0.4, 0.6, 0.2, 0.3]], dtype=np.float32)  # impermanence (anitya)
    
    # Transform using Procrustes
    transformed_procrustes = tf.matmul(new_buddhist_concept, W_procrustes, transpose_b=True)
    
    # Find nearest contemporary concepts
    print("\nNearest contemporary concepts to transformed 'impermanence' (Procrustes):")
    for i, vec in enumerate(contemporary_vectors):
        similarity = np.dot(transformed_procrustes[0], vec) / (
            np.linalg.norm(transformed_procrustes[0]) * np.linalg.norm(vec)
        )
        print(f"Concept {i}: similarity = {similarity:.4f}")
    
    return {
        "procrustes": {
            "W": W_procrustes,
            "aligned_vectors": aligned_vectors_procrustes,
            "avg_similarity": avg_sim_procrustes
        },
        "manhattan": {
            "W": W_manhattan,
            "aligned_vectors": aligned_vectors_manhattan,
            "avg_similarity": avg_sim_manhattan
        }
    }


if __name__ == "__main__":
    results = demonstrate_cross_space_alignment()
    print("\nAlignment results summary:")
    print(f"Procrustes average similarity: {results['procrustes']['avg_similarity']:.4f}")
    print(f"Manhattan average similarity: {results['manhattan']['avg_similarity']:.4f}")