import random

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import spacy

# Load spaCy's medium English model
nlp = spacy.load("en_core_web_md")


# Generate embeddings dynamically with spaCy
def generate_embeddings(words):
    return {word: nlp(word).vector for word in words}


# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Perform dynamic clustering with a distance threshold
def dynamic_clustering(words, word_embeddings, distance_threshold=1.0):
    valid_words = [word for word in words if word in word_embeddings]
    embeddings = np.array([word_embeddings[word] for word in valid_words])

    # Agglomerative clustering with a distance threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Let the algorithm decide based on distance
        metric='euclidean',
        linkage='ward',
        distance_threshold=distance_threshold
    ).fit(embeddings)

    # Group words by cluster
    clusters = {}
    for word, cluster_id in zip(valid_words, clustering.labels_):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(word)

    return clusters


# Generate a hint word for a cluster
def generate_hint_word(cluster_words, word_embeddings):
    cluster_vectors = np.array([word_embeddings[word] for word in cluster_words if word in word_embeddings])
    centroid = np.mean(cluster_vectors, axis=0)

    closest_word = None
    best_score = -np.inf
    candidates = []

    # Use a restricted set of meaningful words
    for word in nlp.vocab:
        if (
            word.has_vector and word.is_alpha and not word.is_stop and
            word.is_lower and len(word.text) > 1 and word.text not in cluster_words
        ):
            # Compute similarity score
            base_similarity = (
                0.7 * cosine_similarity(word.vector, centroid) +
                0.3 * max(cosine_similarity(word.vector, word_embeddings[w]) for w in cluster_words)
            )
            # Apply TF-IDF-inspired weighting
            weighted_score = base_similarity * (1 - abs(word.prob))
            candidates.append((word.text, weighted_score))

            if weighted_score > best_score:
                best_score = weighted_score
                closest_word = word.text

    # Debug: Print candidates
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:10]
    print(f"Top candidates for cluster {cluster_words}: {candidates}")

    # Fallback if no suitable word is found
    if not closest_word:
        fallback_hints = ['action', 'object', 'group', 'event', 'concept']
        closest_word = random.choice(fallback_hints)

    return closest_word






# Main function to generate hints for subclusters
def generate_subcluster_hints(board_words, distance_threshold=1.0):
    # Generate embeddings for the board words
    word_embeddings = generate_embeddings(board_words)

    # Split board words into target and non-target words
    target_words = board_words[:9]
    non_target_words = board_words[9:]

    # Perform dynamic clustering on target words
    subclusters = dynamic_clustering(target_words, word_embeddings, distance_threshold=distance_threshold)

    # Generate a hint for each subcluster
    hints = {}
    for subcluster_id, subcluster_words in subclusters.items():
        hint_word = generate_hint_word(subcluster_words, word_embeddings)
        hints[subcluster_id] = (hint_word, subcluster_words)

    return hints


# Example usage
if __name__ == "__main__":
    # Mock board words
    board_words = [
        "DISEASE", "PARACHUTE", "COPPER", "YARD", "COURT", "SUPERHERO",
        "BUGLE", "SPOT", "BAND", "ARM", "ALPS", "LEMON", "QUEEN", "DRESS",
        "PIPE", "NUT", "UNICORN", "ATLANTIS", "LITTER", "DANCE", "SHIP",
        "WALL", "DRAFT", "LIGHT", "TABLE"
    ]

    # Generate hints for subclusters
    hints = generate_subcluster_hints(board_words, distance_threshold=10)

    # Print the generated hints
    for subcluster_id, (hint, words) in hints.items():
        print(f"Hint for subcluster {subcluster_id}: {hint}")
        print(f"  Words in subcluster: {words}")
