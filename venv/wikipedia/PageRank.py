import json
import random

DAMPING = 0.85
SAMPLES = 10000


def run_pagerank():
    # Open file
    with open('wiki.json', 'r') as json_file:
        data = json.load(json_file)

    corpus = clean(data) 
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    for page in sorted(ranks, key=lambda x: ranks[x]):
        print(f"{page}: {ranks[page]:.5f}")


def clean(data):
    pages = dict()

    # Get page title and links excluding self
    for page in data: 
        pages[page["url"]] = set(page["links"]) - {page["url"]}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )
    return pages


def transition_model(corpus, page, damping_factor):
    prob_dist = dict()

    for p in corpus:
        prob_dist[p] = (1 - damping_factor) / len(corpus)

    # Check if page in corpus and if page has links
    if page in corpus and corpus[page]:
        for link in corpus[page]:
            prob_dist[link] += damping_factor / len(corpus[page]) # Add to probability of clicking link
    else:
        for p in corpus: # Else set even prob dist for all pages
            prob_dist[p] = 1 / len(corpus)

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    page_ranks = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n): # Iterate N times
        dist = transition_model(corpus, current_page, damping_factor)

        for page in corpus:
            page_ranks[page] += dist[page] # Add prob distributions to page ranks

        current_page = random.choices(list(corpus.keys()), weights=list(dist.values()))[0] # Choose random page with given distribution

    total_ranks = sum(page_ranks.values()) # Ranks sum

    for p in page_ranks: # Normalize page ranks
        page_ranks[p] /= total_ranks

    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_ranks = {page: 1/len(corpus) for page in corpus}
    N = len(corpus)
    change = 1

    while change > 0.001:
        new_ranks = {page: 0 for page in corpus}

        for p in corpus:
            linked_page_rank = 0

            for linked_page in corpus:
                if p in corpus[linked_page]:#
                    linked_page_rank += page_ranks[linked_page]/len(corpus[linked_page])

            new_ranks[p] = (1-damping_factor)/N + damping_factor * linked_page_rank

        change = sum(abs(page_ranks[page]-new_ranks[page]) for page in corpus)
        page_ranks = new_ranks

    return page_ranks

run_pagerank()