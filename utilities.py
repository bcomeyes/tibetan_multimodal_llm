"""
utilities.py - Support functions for embedding processing and model evaluation

This module provides utility functions for embedding handling, model evaluation, and 
reference management. It includes:

1. load_pretrained_embeddings(): A function that simulates loading pretrained word
   embeddings with specific handling for domain-specific vocabulary. In a production
   environment, this would connect to actual embedding files like Word2Vec, GloVe,
   or domain-specific embeddings.

2. evaluate_model_performance(): A framework for model evaluation against standard
   metrics (accuracy, F1, precision, recall). This implementation returns mock values
   but provides the interface for actual evaluation pipelines.

3. save_model_checkpoint(): A stub for model persistence functionality, which would
   normally use TensorFlow's checkpoint mechanism in production code.

4. get_references_for_code_implementation(): A function that returns academic references
   relevant to the algorithmic implementations throughout the codebase.

These utilities handle common ML operations like embedding normalization, vocabulary
management, and performance evaluation. The functions are designed to be easily replaced
with production-grade implementations while maintaining consistent interfaces.

Note: While the vocabulary examples include Buddhist terminology, these utility functions
are general-purpose and would work with any domain-specific vocabulary and embedding set.
"""


import numpy as np
import tensorflow as tf

def load_pretrained_embeddings(path, vocab_size=10000, embedding_dim=300):
    """ 
    Utility function to simulate loading pretrained embeddings
    
    Parameters:
    - path: Path to the embeddings file (not actually used in this demo)
    - vocab_size: Size of vocabulary to simulate
    - embedding_dim: Dimension of embeddings

    Returns:
    - embeddings: Numpy array of shape [vocab_size, embedding_dim]
    - word2idx: Dictionary mapping words to indices
    """
    # In a real implementation, we would load actual embeddings from a file
    # Here we generate random embeddings for demonstration
    np.random.seed(42)  # For reproducibility
    embeddings = np.random.normal(size=(vocab_size, embedding_dim))

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Create a dummy vocabulary
    words = [f"word_{i}" for i in range(vocab_size)]
    word2idx = {word: i for i, word in enumerate(words)}

    # Add some Buddhist terms to the vocabulary for demonstration
    buddhist_terms = [
        "emptiness", "impermanence", "non-self", "suffering", "compassion",
        "mindfulness", "meditation", "enlightenment", "buddha", "dharma",
        "sangha", "nirvana", "sunyata", "anicca", "anatta"
    ]

    for i, term in enumerate(buddhist_terms):
        word2idx[term] = i
        # Replace the random embedding with a slightly biased one
        embeddings[i] = np.random.normal(0.2, 0.8, size=(embedding_dim,))
        embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

    return embeddings, word2idx


def evaluate_model_performance(model, test_data, metrics=['accuracy', 'f1']):
    """ 
    Utility function to evaluate model performance
    
    Parameters:
    - model: Trained model
    - test_data: Test dataset
    - metrics: List of metrics to compute

    Returns:
    - results: Dictionary of metric values
    """
    # In a real implementation, we would compute actual metrics
    # Here we return simulated results for demonstration
    results = {
        'accuracy': 0.87,
        'f1': 0.83,
        'precision': 0.85,
        'recall': 0.81
    }

    # Return only requested metrics
    return {metric: results[metric] for metric in metrics if metric in results}


def save_model_checkpoint(model, path):
    """ 
    Utility function to save model checkpoints
    
    Parameters:
    - model: Model to save
    - path: Path to save the model
    """
    # In a real implementation, we would save the model to disk
    # Here we just print a message for demonstration
    print(f"Model saved to {path}")

    # In TensorFlow, we would use:
    # model.save_weights(path)
    return


def get_references_for_code_implementation():
    """ 
    Return a list of academic references relevant to the code implementations 
    """
    references = [
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.",
        "Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26.",
        "Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Representation learning on graphs: Methods and applications. IEEE Data Engineering Bulletin, 40(3), 52-74.",
        "Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. International Conference on Learning Representations.",
        "Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. International conference on machine learning, 1597-1607.",
        "Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and computing, 17(4), 395-416.",
        "Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2019). Multimodal machine learning: A survey and taxonomy. IEEE transactions on pattern analysis and machine intelligence, 41(2), 423-443.",
        "Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., ... & Liang, P. (2022). On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.",
        "Lample, G., Conneau, A., Ranzato, M. A., Denoyer, L., & Jégou, H. (2018). Word translation without parallel data. International Conference on Learning Representations.",
        "Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). Knowledge graph embedding by translating on hyperplanes. In Proceedings of the AAAI Conference on Artificial Intelligence, 28(1).",
        "Trivedi, R., Dai, H., Wang, Y., & Song, L. (2019). Know-evolve: Deep temporal reasoning for dynamic knowledge graphs. In International Conference on Machine Learning, 3462-3471."
    ]
    return references


def demo_usage():
    """
    Demonstrate usage of utility functions
    """
    # Load pretrained embeddings
    embeddings, word2idx = load_pretrained_embeddings("dummy_path.txt", vocab_size=100)
    print(f"Loaded embeddings of shape: {embeddings.shape}")
    print(f"Vocabulary size: {len(word2idx)}")
    
    # Check if Buddhist terms were added
    buddhist_terms = ["emptiness", "impermanence", "non-self"]
    for term in buddhist_terms:
        if term in word2idx:
            print(f"Buddhist term '{term}' found at index {word2idx[term]}")
    
    # Simulate model evaluation
    dummy_model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    metrics = evaluate_model_performance(dummy_model, None, metrics=['accuracy', 'f1'])
    print(f"Model performance: {metrics}")
    
    # Simulate model checkpointing
    save_model_checkpoint(dummy_model, "model_checkpoint.h5")
    
    # Get academic references
    references = get_references_for_code_implementation()
    print(f"Found {len(references)} academic references")
    print(f"First reference: {references[0]}")


if __name__ == "__main__":
    demo_usage()