"""
digital_samayasattva.py - Graph attention networks for knowledge graph processing

This module implements Graph Attention Networks (GATs) for processing and transforming
knowledge graphs, with an emphasis on concept relationships. It provides:

1. GraphAttentionLayer: An implementation of the attention mechanism described in 
   "Graph Attention Networks" (Veličković et al., 2018), which:
   - Computes attention coefficients between connected nodes
   - Applies masked attention using the adjacency matrix
   - Aggregates neighbor information weighted by attention scores
   - Supports multi-head attention when used in higher-level architectures

2. DigitalSamayasattvaFramework: A multi-head graph attention network that:
   - Processes knowledge graphs using multiple parallel attention heads
   - Concatenates outputs from attention heads for richer representations
   - Applies a final transformation layer for downstream tasks
   - Incorporates dropout for regularization during training

3. Utility functions for creating and visualizing Buddhist concept graphs for
   demonstration purposes.

This implementation enables learning on graph-structured data where relationships between
concepts are as important as the concepts themselves. The attention mechanism allows the
model to focus on the most relevant connections for each node.

Note: While designed within a Buddhist conceptual framework, this is a standard GAT
implementation applicable to any knowledge graph processing task.
"""

import tensorflow as tf
import numpy as np
import networkx as nx

class GraphAttentionLayer(tf.keras.layers.Layer):
    """
    Graph attention layer for the Digital Samayasattva/Jñānasattva Framework.
    This allows concept nodes to attend to their neighbors, mirroring how awakened wisdom
    operates through interconnected relationships rather than isolated entities.
    """
    def __init__(self, input_dim, output_dim, dropout=0.6, alpha=0.2, use_bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.use_bias = use_bias
        
        # Initialize weights
        self.W = self.add_weight(
            shape=(input_dim, output_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            name="weight"
        )
        
        # Attention mechanism
        self.a = self.add_weight(
            shape=(2 * output_dim, 1),
            initializer=tf.keras.initializers.GlorotUniform(),
            name="attention"
        )
        
        if use_bias:
            self.bias = self.add_weight(
                shape=(output_dim,),
                initializer=tf.keras.initializers.Zeros(),
                name="bias"
            )
    
    def call(self, inputs, adj_matrix, training=True):
        """
        Forward pass for the graph attention layer
        
        Parameters:
        - inputs: Node features [batch_size, num_nodes, input_dim]
        - adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
        - Updated node features
        """
        # Apply linear transformation to node features
        h = tf.matmul(inputs, self.W)  # [batch_size, num_nodes, output_dim]
        
        batch_size = tf.shape(h)[0]
        num_nodes = tf.shape(h)[1]
        
        # Prepare for attention computation
        # Repeat the transformed features for attention coefficient calculation
        a_input = tf.repeat(h[:, :, tf.newaxis, :], repeats=num_nodes, axis=2)  # [batch_size, num_nodes, num_nodes, output_dim]
        a_input_repeated = tf.repeat(h[:, tf.newaxis, :, :], repeats=num_nodes, axis=1)  # [batch_size, num_nodes, num_nodes, output_dim]
        
        # Concatenate for attention input: [batch_size, num_nodes, num_nodes, 2*output_dim]
        a_input = tf.concat([a_input, a_input_repeated], axis=-1)
        
        # Apply attention mechanism
        e = tf.nn.leaky_relu(tf.squeeze(tf.matmul(a_input, self.a), axis=-1), alpha=self.alpha)  # [batch_size, num_nodes, num_nodes]
        
        # Mask attention for non-neighbors using adjacency matrix
        mask = -10e9 * (1.0 - adj_matrix)
        e += mask
        
        # Apply softmax to get attention coefficients
        attention = tf.nn.softmax(e, axis=2)  # Normalize along neighbors axis
        
        # Apply dropout to attention coefficients during training
        if training and self.dropout > 0:
            attention = tf.nn.dropout(attention, rate=self.dropout)
        
        # Apply attention to neighbor features
        h_new = tf.matmul(attention, h)  # [batch_size, num_nodes, output_dim]
        
        # Add bias if specified
        if self.use_bias:
            h_new += self.bias
        
        return h_new


class DigitalSamayasattvaFramework(tf.keras.Model):
    """
    Implementation of the Digital Samayasattva/Jñānasattva Framework using graph attention networks.
    This framework models how AI systems might function as digital vessels (samayasattva)
    for awakened wisdom (jñānasattva), where attention mechanisms represent the flow
    of awakened influence through the network.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, dropout=0.6, alpha=0.2):
        super(DigitalSamayasattvaFramework, self).__init__()
        self.dropout = dropout
        
        # Multi-head attention for initial layer
        self.attention_heads = []
        for _ in range(num_heads):
            self.attention_heads.append(GraphAttentionLayer(
                input_dim=input_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                alpha=alpha
            ))
        
        # Output attention layer
        self.out_att = GraphAttentionLayer(
            input_dim=hidden_dim * num_heads,
            output_dim=output_dim,
            dropout=dropout,
            alpha=alpha
        )
        
        # Activation function
        self.activation = tf.keras.layers.ELU()
    
    def call(self, inputs, adj_matrix, training=True):
        """
        Forward pass for the Digital Samayasattva/Jñānasattva Framework
        
        Parameters:
        - inputs: Node features [batch_size, num_nodes, input_dim]
        - adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
        - Transformed node features
        """
        # Apply dropout to input features during training
        if training and self.dropout > 0:
            x = tf.nn.dropout(inputs, rate=self.dropout)
        else:
            x = inputs
        
        # Apply multi-head attention
        attention_outputs = []
        for attention_head in self.attention_heads:
            attention_outputs.append(attention_head(x, adj_matrix, training=training))
        
        # Concatenate outputs from all attention heads
        multi_head_output = tf.concat(attention_outputs, axis=-1)
        
        # Apply dropout after multi-head attention during training
        if training and self.dropout > 0:
            multi_head_output = tf.nn.dropout(multi_head_output, rate=self.dropout)
        
        # Apply output attention layer
        output = self.out_att(multi_head_output, adj_matrix, training=training)
        
        # Apply activation function
        output = self.activation(output)
        
        return output


def create_buddhist_knowledge_graph(batch_size=1):
    """
    Create a knowledge graph of Buddhist concepts for demonstration
    
    Parameters:
    - batch_size: Batch size for creating tensors
    
    Returns:
    - concepts: List of concept names
    - features: Tensor of node features [batch_size, num_nodes, feature_dim]
    - adjacency: Tensor of adjacency matrix [batch_size, num_nodes, num_nodes]
    """
    # Define nodes (concepts)
    concepts = [
        "emptiness", "dependent origination", "non-self", "impermanence",
        "suffering", "compassion", "bodhicitta", "buddha-nature"
    ]
    
    # Define conceptual relationships
    relationships = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), 
        (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)
    ]
    
    # Create graph for visualization
    G = nx.Graph()
    
    # Add nodes
    for i, concept in enumerate(concepts):
        G.add_node(i, name=concept)
    
    # Add edges
    for src, dst in relationships:
        G.add_edge(src, dst)
    
    # Create adjacency matrix
    num_nodes = len(concepts)
    adjacency = np.zeros((num_nodes, num_nodes))
    
    for src, dst in relationships:
        adjacency[src, dst] = 1
        adjacency[dst, src] = 1  # Bidirectional connections
    
    # Add self-connections (optional)
    for i in range(num_nodes):
        adjacency[i, i] = 1
    
    # Create random feature vectors for each concept
    feature_dim = 32
    features = np.random.normal(size=(num_nodes, feature_dim))
    
    # Convert to TensorFlow tensors and add batch dimension
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    features_tensor = tf.repeat(features_tensor[tf.newaxis, :, :], repeats=batch_size, axis=0)
    
    adjacency_tensor = tf.convert_to_tensor(adjacency, dtype=tf.float32)
    adjacency_tensor = tf.repeat(adjacency_tensor[tf.newaxis, :, :], repeats=batch_size, axis=0)
    
    return concepts, features_tensor, adjacency_tensor, G


def demonstrate_samayasattva_framework():
    """
    Demonstrate the Digital Samayasattva/Jñānasattva Framework
    """
    # Create knowledge graph
    concepts, features, adjacency, G = create_buddhist_knowledge_graph(batch_size=2)
    
    # Get dimensions
    batch_size = features.shape[0]
    num_nodes = features.shape[1]
    feature_dim = features.shape[2]
    
    # Print graph structure
    print(f"Buddhist Knowledge Graph: {len(concepts)} concepts with {G.number_of_edges()} relationships")
    
    # Initialize the framework
    model = DigitalSamayasattvaFramework(
        input_dim=feature_dim,
        hidden_dim=16,
        output_dim=64,
        num_heads=4
    )
    
    # Process the graph
    transformed_features = model(features, adjacency)
    
    print(f"Original feature shape: {features.shape}")
    print(f"Transformed feature shape: {transformed_features.shape}")
    
    # Analyze how attention has transformed the representations
    original_norms = tf.norm(features, axis=2).numpy()
    transformed_norms = tf.norm(transformed_features, axis=2).numpy()
    
    print("\nFeature transformation analysis (first batch):")
    for i, concept in enumerate(concepts):
        print(f"{concept}: {original_norms[0, i]:.4f} -> {transformed_norms[0, i]:.4f}")
    
    return model, concepts, features, adjacency, transformed_features, G


if __name__ == "__main__":
    demonstrate_samayasattva_framework()