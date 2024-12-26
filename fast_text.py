from gensim.models.keyedvectors import KeyedVectors
import os
import re
import math

# -----------------------
# Optional: WordNet
# -----------------------
try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False


def load_or_save_fasttext_model(vec_filepath, kv_filepath):
    """Loads or saves the fastText model."""
    if os.path.exists(kv_filepath):
        print("Loading fastText model from saved .kv file...")
        model = KeyedVectors.load(kv_filepath)
    else:
        print("Loading fastText vectors from .vec file...")
        model = KeyedVectors.load_word2vec_format(vec_filepath, binary=False, no_header=True)
        print("Saving fastText model as .kv file for faster future loading...")
        model.save(kv_filepath)
    return model

def is_morph_variant(candidate, targets):
    """
    Filters out morphological variants or substrings of the target words.
    e.g. if 'whale' is a target, 'whaler' or 'whales' get removed.
    """
    c_low = candidate.lower()
    for t in targets:
        if t.lower() in c_low and len(t) > 3:  # Avoid substrings shorter than 4 characters
            return True
    return False

def compute_idf(vocab_size, word_rank):
    """
    A crude IDF-like measure:
      - vocab_size is total # of words in model.index_to_key
      - word_rank is the position in index_to_key (0-based)
    """
    return math.log((vocab_size + 1) / (word_rank + 1))

def rarity_boost(idf, alpha=0.1, idf_cap=3.0):
    """
    IDF-based rarity boost.
    """
    clipped = min(idf, idf_cap)
    return 1.0 + alpha * clipped

def combined_score(word, w1, w2, model, vocab_size,
                   alpha=0.1, idf_cap=3.0, balance_beta=2.0):
    """
    Final scoring logic = product of similarities * IDF boost * balance factor
    Returns (final_score, sim1, sim2, idf).
    """
    if word not in model:
        return 0.0, 0.0, 0.0, 0.0

    sim1 = model.similarity(word, w1)
    sim2 = model.similarity(word, w2)
    semantic_score = sim1 * sim2

    rank = model.key_to_index.get(word, vocab_size - 1)
    idf = compute_idf(vocab_size, rank)
    rb = rarity_boost(idf, alpha=alpha, idf_cap=idf_cap)

    diff = abs(sim1 - sim2)
    balance_factor = math.exp(-balance_beta * diff)

    final = semantic_score * rb * balance_factor
    return final, sim1, sim2, idf

# ----------------------------------------------------------------------
#  WordNet-based Hypernym / Meronym Check
# ----------------------------------------------------------------------

def get_all_hypernyms(word):
    """
    Returns a set of all hypernym lemmas for all synsets of 'word'.
    E.g. 'country' might appear as a hypernym for 'Germany'.
    """
    results = set()
    for syn in wn.synsets(word):
        for hyper_chain in syn.hypernym_paths():
            for hyper_syn in hyper_chain:
                # Add all lemma names
                for lemma in hyper_syn.lemmas():
                    results.add(lemma.name().lower())
    return results

def get_all_meronyms(word):
    """
    Returns a set of all meronym lemmas (part-of) for all synsets of 'word'.
    Could be relevant if we want 'wheel' as a meronym of 'car', for instance.
    """
    results = set()
    for syn in wn.synsets(word):
        # part_meronyms = 'part of' relationships
        for part in syn.part_meronyms():
            for lemma in part.lemmas():
                results.add(lemma.name().lower())
        # member_meronyms = members of a group/collection
        for mem in syn.member_meronyms():
            for lemma in mem.lemmas():
                results.add(lemma.name().lower())
        # substance_meronyms = physical components
        for sub in syn.substance_meronyms():
            for lemma in sub.lemmas():
                results.add(lemma.name().lower())
    return results

def gather_wordnet_sets(word, hypernyms=True, meronyms=False):
    """
    Gathers all hypernyms/meronyms for 'word' in WordNet.
    Returns a dictionary with keys 'hypernyms' and 'meronyms'.
    """
    if not WORDNET_AVAILABLE:
        return {"hypernyms": set(), "meronyms": set()}

    hset = get_all_hypernyms(word) if hypernyms else set()
    mset = get_all_meronyms(word) if meronyms else set()
    return {"hypernyms": hset, "meronyms": mset}

def build_shared_wordnet_sets(w1, w2, hypernyms=True, meronyms=False):
    """
    Returns a tuple (shared_hypernyms, shared_meronyms) for both words w1 and w2.
    """
    sets1 = gather_wordnet_sets(w1, hypernyms=hypernyms, meronyms=meronyms)
    sets2 = gather_wordnet_sets(w2, hypernyms=hypernyms, meronyms=meronyms)

    shared_hyper = sets1["hypernyms"].intersection(sets2["hypernyms"])
    shared_mero = sets1["meronyms"].intersection(sets2["meronyms"])
    return (shared_hyper, shared_mero)

def re_rank_hyper_meronym(candidates, w1, w2,
                          model, vocab_size=1_000_000,
                          alpha=0.1, idf_cap=3.0, balance_beta=2.0,
                          hyper_boost=1.2, mero_boost=1.1,
                          debug_logs=False):
    """
    Takes the list of (word, score, sim1, sim2, idf) and looks for
    candidates that are in the 'shared hypernym' or 'shared meronym' sets.
    If found, multiply their existing score by hyper_boost or mero_boost.

    returns a new list, re-sorted by the new score.
    """
    if not WORDNET_AVAILABLE:
        if debug_logs:
            print("[WordNet] Not available, skipping hypernym/meronym check.")
        return candidates

    shared_hyper, shared_mero = build_shared_wordnet_sets(w1, w2,
                                                          hypernyms=True,
                                                          meronyms=True)

    # We make a dictionary for quick membership checks
    # e.g. "country" in shared_hyper => True
    # We'll handle lowercase matching because lemma names are often lowercase.
    shared_hyper_lc = set(x.lower() for x in shared_hyper)
    shared_mero_lc  = set(x.lower() for x in shared_mero)

    # Re-rank logic
    new_candidates = []
    for word, score, s1, s2, idf in candidates:
        base_word_lc = word.lower()
        new_score = score

        # If the candidate is in the shared hypernym set, boost it
        if base_word_lc in shared_hyper_lc:
            new_score *= hyper_boost
            if debug_logs:
                print(f"[Hypernym Boost] '{word}' => x{hyper_boost:.1f}, new_score={new_score:.4f}")

        # If the candidate is in the shared meronym set, boost it (less likely for countries)
        if base_word_lc in shared_mero_lc:
            new_score *= mero_boost
            if debug_logs:
                print(f"[Meronym Boost] '{word}' => x{mero_boost:.1f}, new_score={new_score:.4f}")

        new_candidates.append((word, new_score, s1, s2, idf))

    # Sort by new score
    new_candidates.sort(key=lambda x: x[1], reverse=True)
    return new_candidates

# ----------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------
def find_hints_with_rarity(
    w1,
    w2,
    model,
    forbidden_words=None,
    topn=1000,
    alpha=0.1,
    idf_cap=3.0,
    balance_beta=2.0,
    debug_logs=False,
    use_wordnet_hypernym=False,
    hyper_boost=1.2,
    mero_boost=1.1
):
    """
    Returns top hints for (w1, w2) using:
      - "rarity + balance" combined score
      - optional WordNet hypernym/meronym second-stage re-ranking

    :param use_wordnet_hypernym: if True, we apply re_rank_hyper_meronym
                                 to push up shared hypernyms/meronyms.
    """
    vocab_size = 1_000_000  # for 'wiki-news-300d-1M-subword'

    if debug_logs:
        print(f"[INFO] Generating top candidates for '{w1}' and '{w2}'.")
        print(f"       alpha={alpha}, idf_cap={idf_cap}, balance_beta={balance_beta}, topn={topn}")
        if WORDNET_AVAILABLE:
            print("[INFO] WordNet is available.")
        else:
            print("[WARNING] WordNet not found. Hypernym/meronym step will be skipped if enabled.")

    forbidden_words = set(forbidden_words or [])
    candidates = model.index_to_key[:topn]

    scored_candidates = []
    for w in candidates:
        if w.lower() in forbidden_words:
            continue
        if is_morph_variant(w, [w1, w2]):
            continue

        score, sim1, sim2, idf = combined_score(
            w, w1, w2, model, vocab_size,
            alpha=alpha,
            idf_cap=idf_cap,
            balance_beta=balance_beta
        )
        scored_candidates.append((w, score, sim1, sim2, idf))

    # Sort descending
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    if debug_logs:
        print("\n[DEBUG] === Pre-WordNet/Re-rank Top 10 ===")
        for candidate in scored_candidates[:10]:
            w, s, s1, s2, i = candidate
            print(f"   {w} | score={s:.4f}, sim1={s1:.4f}, sim2={s2:.4f}, idf={i:.4f}")

    # Optional: re-rank with hypernym/meronym check
    if use_wordnet_hypernym:
        if debug_logs:
            print("\n[DEBUG] === Applying hypernym/meronym re-rank ===")
        scored_candidates = re_rank_hyper_meronym(
            scored_candidates,
            w1, w2,
            model=model,
            vocab_size=vocab_size,
            alpha=alpha,
            idf_cap=idf_cap,
            balance_beta=balance_beta,
            hyper_boost=hyper_boost,
            mero_boost=mero_boost,
            debug_logs=debug_logs
        )

        if debug_logs:
            print("\n[DEBUG] === Post-WordNet/Re-rank Top 10 ===")
            for candidate in scored_candidates[:10]:
                w, s, s1, s2, i = candidate
                print(f"   {w} | score={s:.4f}, sim1={s1:.4f}, sim2={s2:.4f}, idf={i:.4f}")

    # Return top 50
    return scored_candidates[:50]


if __name__ == "__main__":
    # Example usage
    vec_filepath = "wiki-news-300d-1M-subword.vec"  # Adjust path
    kv_filepath = "fasttext_model.kv"
    fasttext_model = load_or_save_fasttext_model(vec_filepath, kv_filepath)

    input_words = ["chocolate", "mint"]
    forbidden_words = {"japan", "tokyo", "city", "asia"}

    hints_with_scores = find_hints_with_rarity(
        w1=input_words[0],  # First target word
        w2=input_words[1],  # Second target word
        model=fasttext_model,  # The loaded FastText KeyedVectors model
        forbidden_words=forbidden_words,
        topn=100000,  # Number of vocabulary words to scan before scoring and ranking
        alpha=0.2,  # Weighting factor for IDF-based rarity boost (higher = bigger boost for rare words)
        idf_cap=5.0,  # Maximum cap on the IDF boost (prevents extremely rare words from dominating)
        balance_beta=1.5,  # Controls how strongly we penalize lopsided similarity to the two target words
        # (higher = more penalty when sim1 and sim2 differ)
        debug_logs=True,  # Whether to print debug/logging info for the pipeline steps
        use_wordnet_hypernym=True,  # Enables second-stage re-ranking based on WordNet hypernym/meronym checks
        hyper_boost=2,  # Multiplicative factor applied to a candidate if it's a shared hypernym of both targets
        mero_boost=1.5  # Multiplicative factor if it's a shared meronym of both targets
    )

    print("\n=== Final Top Hints ===")
    for hint, c_score, s1, s2, idf in hints_with_scores:
        print(f"{hint} (score={c_score:.4f}, sim1={s1:.4f}, sim2={s2:.4f}, idf={idf:.4f})")
