import numpy as np
from annoy import AnnoyIndex


def build_annoy_index(word_embeddings, dim=300, n_trees=10):
    """
    Build an Annoy index for fast approximate nearest neighbor search.
    word_embeddings: dict { word: np.array(dim) }
    dim: embedding dimension
    n_trees: number of trees to build (trade-off between speed and accuracy)
    """
    # Create Annoy index
    t = AnnoyIndex(dim, metric='angular')  # 'angular' is basically cosine distance

    # Keep a mapping from index -> word
    words = list(word_embeddings.keys())

    for i, w in enumerate(words):
        vec = word_embeddings[w]
        t.add_item(i, vec)

    t.build(n_trees)
    return t, words


def target_vector_2(word_embeddings, target_word1, target_word2):
    """
    Returns a single embedding representing the combination of two target words.
    For simplicity, we average their vectors if both exist.
    """
    if target_word1 not in word_embeddings or target_word2 not in word_embeddings:
        raise ValueError("Both target words must exist in the embedding dictionary.")

    return (word_embeddings[target_word1] + word_embeddings[target_word2]) / 2.0


def get_top_candidates(target_vec, annoy_index, words, top_k=50):
    """
    Use the Annoy index to get the top_k nearest neighbors to target_vec.
    Returns a list of (word, distance) pairs.
    """
    # Add the target vector temporarily to the index
    temp_index = len(words)
    annoy_index.add_item(temp_index, target_vec)
    annoy_index.build(1)  # Add a single tree for the temporary vector
    neighbors = annoy_index.get_nns_by_item(temp_index, top_k, include_distances=True)
    annoy_index.unbuild()  # Remove the temporary vector from the index
    return [(words[i], dist) for i, dist in zip(neighbors[0], neighbors[1])]


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def re_score_top_candidates(top_candidates, word_embeddings, target_word1, target_word2):
    """
    Among a small set of top_candidates, re-score each by sum of similarity to the target words.
    Return the best word and its score.
    """
    best_hint, best_score = None, -np.inf
    for (w, dist) in top_candidates:
        # Skip if the word is the same as either target word
        if w == target_word1 or w == target_word2:
            continue

        score = 0.0
        if target_word1 in word_embeddings:
            score += cosine_similarity(word_embeddings[w], word_embeddings[target_word1])
        if target_word2 in word_embeddings:
            score += cosine_similarity(word_embeddings[w], word_embeddings[target_word2])

        if score > best_score:
            best_score = score
            best_hint = w

    return best_hint, best_score


def find_best_hint_for_two_fast(target_word1, target_word2, word_embeddings, top_k=50):
    """
    Find the best hint word by:
    1) Building an approximate search index (Annoy) over word_embeddings
    2) Forming a target vector (average of the two target words)
    3) Retrieving top_k approximate neighbors
    4) Re-scoring those neighbors by sum of similarity to both target words
    """
    # Build Annoy index
    dim = len(next(iter(word_embeddings.values())))  # e.g., 300
    annoy_index, words = build_annoy_index(word_embeddings, dim=dim, n_trees=10)

    # Create the target vector
    target_vec = target_vector_2(word_embeddings, target_word1, target_word2)

    # Retrieve top_k candidates (approx nearest neighbors) to the target vector
    top_candidates = get_top_candidates(target_vec, annoy_index, words, top_k)

    # Re-score those candidates
    best_hint, best_score = re_score_top_candidates(top_candidates, word_embeddings, target_word1, target_word2)
    return best_hint, best_score


# Example Usage:
if __name__ == "__main__":
    # Example word embeddings
    def create_random_embeddings(vocab, dim=300):
        """
        Create random embeddings for demonstration purposes.
        """
        return {word: np.random.randn(dim) for word in vocab}

    # Example Vocabulary
    vocabulary = ["cat", "dog", "animal", "pet", "chair", "banana", "car", "fruit"]

    # Create random embeddings
    word_embeddings = create_random_embeddings(vocabulary)

    # Example target words
    target_word1 = "cat"
    target_word2 = "dog"

    # Run the hint finder
    best_hint, best_score = find_best_hint_for_two_fast(target_word1, target_word2, word_embeddings, top_k=3)
    print(f"Best hint: {best_hint}, Score: {best_score}")
