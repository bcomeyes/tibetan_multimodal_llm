"""
cross_traditional_connections.py - Spectral clustering for concept graph analysis

This module implements spectral clustering techniques for identifying related concept
clusters in weighted graphs. It provides:

1. CrossTraditionalClusteringModel: A clustering model that:
   - Computes the normalized graph Laplacian from an adjacency matrix
   - Applies spectral clustering to identify concept communities
   - Analyzes cluster composition across different categorical attributes
   - Supports parameterized cluster count

2. Graph creation and visualization utilities:
   - Functions for constructing weighted concept graphs
   - Methods for building adjacency matrices from relationship dictionaries
   - Visualization tools for displaying clustered graphs with categorical attributes

The implementation leverages scikit-learn's SpectralClustering with custom pre-processing
and post-processing to handle weighted edges and analyze cluster composition. The normalized
Laplacian approach helps identify meaningful communities based on connection patterns rather
than just connection density.

This technique is particularly useful for discovering latent structures in knowledge graphs
where concepts may be related across different categorizations or traditions. It can reveal
underlying similarities that aren't apparent from categorical labels alone.

Note: While applied to Buddhist traditions, spectral clustering is a general graph analysis
technique widely used in community detection, image segmentation, and recommendation systems.
"""

import tensorflow as tf
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

class CrossTraditionalClusteringModel:
    """
    Model for identifying clusters of concepts that span different Buddhist traditions
    using spectral clustering on a concept relationship graph.
    This approach reveals underlying unity in diverse expressions of Buddhist wisdom.
    """
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.clustering = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels="discretize",
            random_state=42,
            n_init=100
        )
    
    def compute_graph_laplacian(self, adjacency):
        """
        Compute the normalized graph Laplacian
        
        Parameters:
        - adjacency: Adjacency matrix of the concept graph
        
        Returns:
        - laplacian_sym: Normalized graph Laplacian
        """
        # Convert to numpy if it's a TensorFlow tensor
        if isinstance(adjacency, tf.Tensor):
            adjacency = adjacency.numpy()
        
        # Compute degree matrix
        degree = np.diag(np.sum(adjacency, axis=1))
        
        # Compute Laplacian
        laplacian = degree - adjacency
        
        # Avoid division by zero by adding a small constant to diagonal
        degree_inv_sqrt = np.linalg.inv(np.sqrt(degree + 1e-10 * np.eye(degree.shape[0])))
        
        # Compute normalized Laplacian
        laplacian_sym = degree_inv_sqrt @ laplacian @ degree_inv_sqrt
        
        return laplacian_sym
    
    def identify_clusters(self, adjacency):
        """
        Identify clusters using spectral clustering
        
        Parameters:
        - adjacency: Adjacency matrix representing concept relationships
        
        Returns:
        - labels: Cluster assignment for each concept
        """
        # Compute the normalized Laplacian
        laplacian_sym = self.compute_graph_laplacian(adjacency)
        
        # Use adjacency as affinity matrix for clustering
        self.clustering.affinity = 'precomputed'
        
        # Convert adjacency to affinity (similarity) matrix
        # Higher edge weight means higher similarity
        affinity = adjacency.copy()
        
        # Perform spectral clustering
        labels = self.clustering.fit_predict(affinity)
        
        return labels
    
    def analyze_clusters(self, labels, concepts, traditions):
        """
        Analyze the composition of identified clusters
        
        Parameters:
        - labels: Cluster labels for each concept
        - concepts: List of concept names
        - traditions: Dictionary mapping concepts to their traditions
        
        Returns:
        - clusters: List of lists containing concepts in each cluster
        - analysis: Dictionary with cluster analysis
        """
        # Organize concepts by cluster
        clusters = [[] for _ in range(self.n_clusters)]
        for i, cluster_id in enumerate(labels):
            concept = concepts[i]
            tradition = traditions[concept]
            clusters[cluster_id].append((concept, tradition))
        
        # Analyze clusters
        analysis = []
        for i, cluster in enumerate(clusters):
            traditions_in_cluster = set(tradition for _, tradition in cluster)
            concepts_by_tradition = {}
            for concept, tradition in cluster:
                if tradition not in concepts_by_tradition:
                    concepts_by_tradition[tradition] = []
                concepts_by_tradition[tradition].append(concept)
            
            cluster_analysis = {
                "cluster_id": i,
                "num_traditions": len(traditions_in_cluster),
                "traditions": list(traditions_in_cluster),
                "concepts_by_tradition": concepts_by_tradition,
                "total_concepts": len(cluster)
            }
            analysis.append(cluster_analysis)
        
        return clusters, analysis


def create_cross_traditional_network():
    """
    Create a network of concepts spanning different Buddhist traditions
    
    Returns:
    - concepts_and_traditions: Dictionary mapping concepts to traditions
    - concept_relationships: Dictionary of edge weights between concepts
    """
    # Sample data spanning different Buddhist traditions
    concepts_and_traditions = {
        # Theravada concepts
        "anicca": "Theravada",
        "dukkha": "Theravada",
        "anatta": "Theravada",
        "satipatthana": "Theravada",
        "nibbana": "Theravada",
        
        # Mahayana concepts
        "sunyata": "Mahayana",
        "bodhicitta": "Mahayana",
        "upaya": "Mahayana",
        "tathata": "Mahayana",
        "buddha-nature": "Mahayana",
        
        # Vajrayana concepts
        "deity-yoga": "Vajrayana",
        "mandala": "Vajrayana",
        "clear-light": "Vajrayana",
        "mahamudra": "Vajrayana",
        "rigpa": "Vajrayana",
        
        # Zen concepts
        "zazen": "Zen",
        "mushin": "Zen",
        "koan": "Zen",
        "shikantaza": "Zen",
        "satori": "Zen"
    }
    
    # Sample relationships between concepts with weights
    # Higher weight indicates stronger conceptual relationship
    concept_relationships = {
        ("anicca", "sunyata"): 0.7,        # Impermanence and emptiness
        ("anatta", "sunyata"): 0.8,         # No-self and emptiness
        ("nibbana", "tathata"): 0.6,        # Nirvana and suchness
        ("satipatthana", "zazen"): 0.5,     # Mindfulness and sitting meditation
        ("dukkha", "bodhicitta"): 0.4,      # Suffering and awakening mind
        ("sunyata", "mahamudra"): 0.7,      # Emptiness and great seal
        ("tathata", "rigpa"): 0.6,          # Suchness and pure awareness
        ("buddha-nature", "rigpa"): 0.8,    # Buddha nature and awareness
        ("mushin", "clear-light"): 0.7,     # No-mind and clear light
        ("zazen", "shikantaza"): 0.9,       # Sitting and just sitting
        ("koan", "upaya"): 0.5,             # Koan practice and skillful means
        ("deity-yoga", "mandala"): 0.8,     # Deity practice and mandalas
        ("satori", "nibbana"): 0.6,         # Enlightenment experiences
        ("anatta", "mushin"): 0.7,          # No-self and no-mind
        ("bodhicitta", "deity-yoga"): 0.6,  # Awakening mind and deity practice
    }
    
    return concepts_and_traditions, concept_relationships


def build_adjacency_matrix(concepts, relationships):
    """
    Build adjacency matrix from concepts and relationships
    
    Parameters:
    - concepts: List of concept names
    - relationships: Dictionary of edge weights between concepts
    
    Returns:
    - adjacency: Adjacency matrix with weights
    """
    num_concepts = len(concepts)
    concept_indices = {concept: i for i, concept in enumerate(concepts)}
    
    # Initialize adjacency matrix
    adjacency = np.zeros((num_concepts, num_concepts))
    
    # Add weighted edges
    for (concept_a, concept_b), weight in relationships.items():
        i = concept_indices[concept_a]
        j = concept_indices[concept_b]
        adjacency[i, j] = weight
        adjacency[j, i] = weight  # Make symmetric
    
    return adjacency


def visualize_clusters(G, cluster_labels, tradition_mapping):
    """
    Visualize the clusters of concepts across traditions
    
    Parameters:
    - G: NetworkX graph
    - cluster_labels: Dictionary mapping node names to cluster IDs
    - tradition_mapping: Dictionary mapping node names to traditions
    
    Returns:
    - matplotlib figure
    """
    plt.figure(figsize=(12, 10))
    
    # Define colors and markers
    traditions = set(tradition_mapping.values())
    clusters = set(cluster_labels.values())
    
    tradition_colors = {
        "Theravada": "blue",
        "Mahayana": "green",
        "Vajrayana": "red",
        "Zen": "purple"
    }
    
    cluster_markers = {i: marker for i, marker in zip(clusters, ['o', 's', '^', 'D', 'v', '<', '>', 'p'])}
    
    # Set node attributes
    nx.set_node_attributes(G, cluster_labels, 'cluster')
    nx.set_node_attributes(G, tradition_mapping, 'tradition')
    
    # Calculate layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes by cluster and tradition
    for cluster_id in clusters:
        for tradition in traditions:
            # Select nodes in this cluster and tradition
            nodes = [n for n in G.nodes if G.nodes[n]['cluster'] == cluster_id and
                                          G.nodes[n]['tradition'] == tradition]
            if nodes:
                nx.draw_networkx_nodes(
                    G, pos, 
                    nodelist=nodes,
                    node_color=tradition_colors[tradition],
                    node_shape=cluster_markers[cluster_id],
                    node_size=500,
                    alpha=0.8,
                    label=f"{tradition} (Cluster {cluster_id+1})"
                )
    
    # Draw edges with varying width based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Add legend
    plt.legend(loc='upper right')
    plt.title("Cross-Traditional Concept Clusters in Buddhist Wisdom")
    plt.axis('off')
    plt.tight_layout()
    
    return plt


def demonstrate_cross_traditional_connections():
    """
    Demonstrate the identification of cross-traditional connections
    """
    # Create cross-traditional network
    concepts_and_traditions, concept_relationships = create_cross_traditional_network()
    
    # Get list of all concepts
    concepts = list(concepts_and_traditions.keys())
    
    # Build adjacency matrix
    adjacency = build_adjacency_matrix(concepts, concept_relationships)
    
    # Create model and identify clusters
    model = CrossTraditionalClusteringModel(n_clusters=4)
    cluster_labels = model.identify_clusters(adjacency)
    
    # Analyze clusters
    clusters, analysis = model.analyze_clusters(
        cluster_labels, concepts, concepts_and_traditions
    )
    
    # Create graph for visualization
    G = nx.Graph()
    
    # Add nodes with attributes
    for i, concept in enumerate(concepts):
        G.add_node(
            concept, 
            tradition=concepts_and_traditions[concept],
            cluster=cluster_labels[i]
        )
    
    # Add edges
    for (concept_a, concept_b), weight in concept_relationships.items():
        G.add_edge(concept_a, concept_b, weight=weight)
    
    # Print cluster analysis
    print("Identified clusters spanning traditions:")
    for cluster_info in analysis:
        print(f"Cluster {cluster_info['cluster_id']+1}: Spans {cluster_info['num_traditions']} traditions ({', '.join(cluster_info['traditions'])})")
        for tradition, tradition_concepts in cluster_info['concepts_by_tradition'].items():
            print(f"  {tradition}: {', '.join(tradition_concepts)}")
    
    # Create cluster labels dictionary for visualization
    cluster_labels_dict = {concepts[i]: label for i, label in enumerate(cluster_labels)}
    
    # Visualize the clusters
    plt = visualize_clusters(G, cluster_labels_dict, concepts_and_traditions)
    
    return model, G, clusters, analysis, plt


if __name__ == "__main__":
    demonstrate_cross_traditional_connections()"