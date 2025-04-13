"""
conceptual_blending.py - Vector space operations for exploring latent semantic relationships

This module implements techniques for exploring and generating new conceptual relationships 
through vector space operations on word/concept embeddings. It provides:

1. ConceptualBlending: A class that performs vector manipulations in normalized embedding
   spaces, including:
   
   - analogical_reasoning(): Implements the classic A:B::C:? relationship discovery through
     vector arithmetic (vec_B - vec_A + vec_C), similar to the "king - man + woman = queen"
     operations demonstrated in word2vec. Returns top-k nearest neighbors to the resulting
     vector.
   
   - concept_interpolation(): Creates a spectrum of intermediate concepts by linear 
     interpolation between embedding vectors, allowing exploration of the semantic space
     between established concepts.
   
   - cross_domain_projection(): Projects concepts from one domain to another using a 
     pre-computed alignment matrix, facilitating cross-domain concept mapping.

These techniques enable exploration of latent relationships in embedding spaces and can
generate novel insights not explicitly encoded in the training data. The implementation
supports operations across different embedding spaces when used with alignment matrices
from cross_space_alignment.py.

Note: While framed within the context of exploring Buddhist concepts and their relationship
to contemporary understanding, these techniques are generalizable to any domain where
semantic exploration is valuable.
"""

import tensorflow as tf
import numpy as np

class ConceptualBlending:
    """
    Implements vector space operations for generating new conceptual blends.
    This enables the exploration of latent conceptual spaces between established concepts,
    potentially revealing insights that connect Buddhist traditions with contemporary understanding.
    """
    def __init__(self, buddhist_embeddings, contemporary_embeddings, 
                 buddhist_vocab, contemporary_vocab):
        """
        Initialize with embedding matrices and vocabularies
        
        Parameters:
        - buddhist_embeddings: Matrix of shape [n_buddhist_concepts, embedding_dim]
        - contemporary_embeddings: Matrix of shape [n_contemporary_concepts, embedding_dim]
        - buddhist_vocab: List of Buddhist concept names
        - contemporary_vocab: List of contemporary concept names
        """
        # Convert inputs to TensorFlow tensors if they're not already
        self.buddhist_emb = tf.convert_to_tensor(buddhist_embeddings, dtype=tf.float32)
        self.contemporary_emb = tf.convert_to_tensor(contemporary_embeddings, dtype=tf.float32)
        self.buddhist_vocab = buddhist_vocab
        self.contemporary_vocab = contemporary_vocab
        
        # Normalize embeddings for cosine similarity computations
        self.buddhist_emb_norm = tf.nn.l2_normalize(self.buddhist_emb, axis=1)
        self.contemporary_emb_norm = tf.nn.l2_normalize(self.contemporary_emb, axis=1)
    
    def analogical_reasoning(self, buddhist_concept_a, buddhist_concept_b, 
                            contemporary_concept, top_k=5):
        """
        Perform Buddhist concept A is to Buddhist concept B 
        as Contemporary concept is to what?
        
        This operation reveals new insights by applying the relationship between
        Buddhist concepts to contemporary domains.
        
        Parameters:
        - buddhist_concept_a: First Buddhist concept
        - buddhist_concept_b: Second Buddhist concept
        - contemporary_concept: Contemporary concept
        - top_k: Number of nearest results to return
        
        Returns:
        - List of (concept, similarity) tuples
        """
        # Get indices
        idx_a = self.buddhist_vocab.index(buddhist_concept_a)
        idx_b = self.buddhist_vocab.index(buddhist_concept_b)
        idx_c = self.contemporary_vocab.index(contemporary_concept)
        
        # Get embeddings
        emb_a = self.buddhist_emb[idx_a]
        emb_b = self.buddhist_emb[idx_b]
        emb_c = self.contemporary_emb[idx_c]
        
        # Perform vector operation: c + (b - a)
        # This applies the relationship between a and b to concept c
        result_vector = emb_c + (emb_b - emb_a)
        result_vector_norm = tf.nn.l2_normalize(result_vector, axis=0)
        
        # Compute similarities with all contemporary concepts
        similarities = tf.matmul(
            result_vector_norm[tf.newaxis, :], 
            self.contemporary_emb_norm, 
            transpose_b=True
        )[0]
        
        # Find top-k matches
        values, indices = tf.math.top_k(similarities, k=top_k)
        
        # Return nearest concepts
        return [(self.contemporary_vocab[idx.numpy()], val.numpy()) 
                for idx, val in zip(indices, values)]
    
    def concept_interpolation(self, concept_a, concept_b, steps=5):
        """
        Create intermediate concepts between two concepts
        
        This operation reveals the conceptual spectrum between established concepts,
        showing how they can blend into each other.
        
        Parameters:
        - concept_a: First concept (Buddhist or contemporary)
        - concept_b: Second concept (Buddhist or contemporary)
        - steps: Number of interpolation steps
        
        Returns:
        - List of (alpha, nearest_concepts) tuples
        """
        # Determine which embedding space each concept belongs to
        a_is_buddhist = concept_a in self.buddhist_vocab
        b_is_buddhist = concept_b in self.buddhist_vocab
        
        # Get indices
        idx_a = self.buddhist_vocab.index(concept_a) if a_is_buddhist else self.contemporary_vocab.index(concept_a)
        idx_b = self.buddhist_vocab.index(concept_b) if b_is_buddhist else self.contemporary_vocab.index(concept_b)
        
        # Get embeddings
        emb_a = self.buddhist_emb[idx_a] if a_is_buddhist else self.contemporary_emb[idx_a]
        emb_b = self.buddhist_emb[idx_b] if b_is_buddhist else self.contemporary_emb[idx_b]
        
        # Create interpolations
        interpolations = []
        for step in range(steps + 1):
            alpha = step / steps
            interp_vector = alpha * emb_a + (1 - alpha) * emb_b
            interp_vector_norm = tf.nn.l2_normalize(interp_vector, axis=0)
            
            # Find nearest concept in appropriate space
            if a_is_buddhist and b_is_buddhist:
                # Both concepts are Buddhist, search in Buddhist space
                similarities = tf.matmul(
                    interp_vector_norm[tf.newaxis, :], 
                    self.buddhist_emb_norm, 
                    transpose_b=True
                )[0]
                
                values, indices = tf.math.top_k(similarities, k=3)
                nearest = [(self.buddhist_vocab[idx.numpy()], val.numpy()) 
                          for idx, val in zip(indices, values)]
            else:
                # At least one concept is contemporary, search in contemporary space
                similarities = tf.matmul(
                    interp_vector_norm[tf.newaxis, :], 
                    self.contemporary_emb_norm, 
                    transpose_b=True
                )[0]
                
                values, indices = tf.math.top_k(similarities, k=3)
                nearest = [(self.contemporary_vocab[idx.numpy()], val.numpy()) 
                          for idx, val in zip(indices, values)]
            
            interpolations.append((alpha, nearest))
        
        return interpolations
    
    def cross_domain_projection(self, buddhist_concept, alignment_matrix):
        """
        Project a Buddhist concept into the contemporary space using
        a pre-computed alignment matrix
        
        Parameters:
        - buddhist_concept: Name of Buddhist concept
        - alignment_matrix: Matrix for aligning Buddhist to contemporary space
        
        Returns:
        - List of (concept, similarity) tuples
        """
        # Get index and embedding
        idx = self.buddhist_vocab.index(buddhist_concept)
        emb = self.buddhist_emb[idx]
        
        # Project to contemporary space
        projected = tf.matmul(emb[tf.newaxis, :], alignment_matrix)
        projected_norm = tf.nn.l2_normalize(projected, axis=1)
        
        # Compute similarities with all contemporary concepts
        similarities = tf.matmul(
            projected_norm, 
            self.contemporary_emb_norm, 
            transpose_b=True
        )[0]
        
        # Find top matches
        values, indices = tf.math.top_k(similarities, k=5)
        
        # Return nearest concepts
        return [(self.contemporary_vocab[idx.numpy()], val.numpy()) 
                for idx, val in zip(indices, values)]


def demonstrate_conceptual_blending():
    """
    Demonstrate the conceptual blending techniques
    """
    # Sample vocabularies
    buddhist_vocab = ["emptiness", "impermanence", "non-self", "suffering", "compassion"]
    contemporary_vocab = ["quantum field", "process philosophy", "consciousness", "neuroplasticity", "empathy"]
    
    # Create random embeddings for demonstration
    embedding_dim = 128
    buddhist_emb = tf.random.normal((len(buddhist_vocab), embedding_dim))
    contemporary_emb = tf.random.normal((len(contemporary_vocab), embedding_dim))
    
    # Create blending object
    blender = ConceptualBlending(buddhist_emb, contemporary_emb, buddhist_vocab, contemporary_vocab)
    
    # Analogical reasoning example
    # "emptiness is to non-self as quantum field is to what?"
    results = blender.analogical_reasoning("emptiness", "non-self", "quantum field")
    print("Emptiness is to non-self as quantum field is to:")
    for concept, similarity in results:
        print(f"  {concept} (similarity: {similarity:.4f})")
    
    # Interpolation example
    # Create a spectrum between "compassion" and "empathy"
    interpolations = blender.concept_interpolation("compassion", "empathy", steps=3)
    print("\nInterpolation between compassion and empathy:")
    for alpha, nearest in interpolations:
        concept, similarity = nearest[0]  # Get top match
        print(f"  Alpha={alpha:.2f}: {concept} (similarity: {similarity:.4f})")
    
    # In a real application, we would also demonstrate cross-domain projection
    # using an alignment matrix computed as in the previous section