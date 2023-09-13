from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import math
import string
import pickle
import random
import sys
import os

current_directory = os.path.dirname(__file__)  # Get the directory of the current script
json_file_path = os.path.join(current_directory, '..', 'venv', 'wikipedia', 'wiki.json')


class Index:
    def __init__(self):
        self.data = self.add_documents() # Opened data
        self.word_buckets = dict() # word -> documents containing word
        self.doc_index = dict() # document -> {word -> frequency for word in document words}
        self.page_ranks = self.get_page_ranks()


    def search(self, query):
        print("\n\nSearching...\n")
        query = self.tokenize(query)

        documents = self.get_buckets(query) # Relevant word buckets
        idfs = self.get_idfs(documents, query) # Relevant query-word idfs
        sorted_files = self.sort_documents(documents, idfs, query) # Documents sorted by tf-idf then PageRank
        
        return self.clean_results(sorted_files)
    
    
    def sort_documents(self, documents, idfs, query):
        scores = {document: (0, 0) for document in documents}

        for word in query:

            for document in documents:
                # Title word matches
                if self.data[document]["title"]:
                    title_words = self.tokenize(self.data[document]["title"])
                title_matches = self.word_matches(title_words, word) if title_words else 0

                # Calculate tf-idf score 
                tf = self.doc_index[document].get(word, 0) / self.doc_index[document]["word_count"] # Query term density calculation
                document_score = tf * idfs[word]

                scores[document] = (scores[document][0] + title_matches, scores[document][1] + document_score) 
        
        scores = sorted(scores.keys(), key=lambda x: (-scores[x][0], -scores[x][1]))[:50] # Sort by tf-idf

        print("Sorted documents...")

        return sorted(scores, key=lambda x: -self.page_ranks[x])[:10] # Sort by PageRank
    

    def word_matches(self, document_words, word):
        count = 0
        for document_word in document_words:
            if document_word in word or word in document_word:
                count += 1
        
        return count

        
    def get_idfs(self, documents, query):
        idfs = dict()

        if len(query) == 1: # If one word query return idf -> 1
            idfs[query[0]] = 1
            return idfs

        for word in query:
            word_freq = sum(1 for document in documents if word in self.doc_index[document])
            idf = math.log(len(documents) / word_freq)
            idfs[word] = idf

        print("Calculated idfs...")
        
        return idfs


    def get_buckets(self, query):
        search_documents = set()

        for word in query:
            if word in self.word_buckets:
                search_documents.update(self.word_buckets[word])
        
        print("Searched word buckets...")

        return search_documents


    def create_indexes(self):
        file_words = { # Create url -> tokens dict
            filename: self.tokenize(self.data[filename]["content"])
            for filename in self.data
        } 
        print("Tokenized file words...")

        for file, words in file_words.items():
            for word in words:
                if word in self.word_buckets:
                    self.word_buckets[word].add(file) # Add file to word bucket if bucket exists
                else:
                    self.word_buckets[word] = set([file]) # Create word bucket and add file to set

        print("Created buckets...")

        self.create_doc_index(file_words)


    def create_doc_index(self, documents): # document -> {word -> frequency for word in document}
        for document, words in documents.items():
            self.doc_index[document] = dict() # Initialize dictionary
            self.doc_index[document]["word_count"] = len(words) # Add word count key -> value
            word_counts = Counter(words)

            for word, count in word_counts.items():
                self.doc_index[document][word] = count

        print("Created word indexes...")

        self.save_data()


    def load_data(self):
        path = os.path.join(current_directory, 'data', 'search_data.pkl')

        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.word_buckets = data.get("word_buckets", {})
            self.doc_index = data.get("doc_index", {})

            print("Loading files...")


    def save_data(self):
        path = os.path.join(current_directory, 'data', 'search_data.pkl')

        with open(path, 'wb') as file:
            data = {
                "word_buckets": self.word_buckets,
                "doc_index": self.doc_index,
            }

            pickle.dump(data, file)

            print("Saved word buckets and document indexes...")


    def get_page_ranks(self):
        damping_factor = 0.85

        page_ranks = {page: 0 for page in self.data}
        current_page = random.choice(list(self.data.keys()))

        # Sampling Page Ranks
        for _ in range(1000):
            dist = self.transition_model(self.data, current_page, damping_factor)

            for page in self.data:
                page_ranks[page] += dist[page] # Add prob distributions to page ranks

            current_page = random.choices(list(dist.keys()), weights=list(dist.values()))[0] # Choose random page with given distribution

        total_ranks = sum(page_ranks.values()) # Ranks sum

        for p in page_ranks: # Normalize page ranks
            page_ranks[p] /= total_ranks

        print("Calculated PageRanks...")

        return page_ranks


    def transition_model(self, corpus, page, damping_factor):  
        prob_dist = dict()
        len_corpus =len(corpus)

        for p in corpus:
            prob_dist[p] = (1 - damping_factor) / len_corpus

        # Check if page in corpus and if page has links
        if page in corpus and corpus[page]["links"]:
            for link in corpus[page]["links"]:
                prob_dist[link] += damping_factor / len(corpus[page]) # Add to probability of clicking link
        else:
            for p in corpus: # Else set even prob dist for all pages
                prob_dist[p] = 1 / len_corpus

        return prob_dist
        

    def tokenize(self, document):
        tokens = nltk.word_tokenize(document.lower())

        words = [
            token for token in tokens
            if token not in stopwords.words('english') and # Remove determinants etc.
            token not in string.punctuation # Remove punctuation
        ]

        return words
    

    def add_documents(self):
        with open(json_file_path, 'r') as json_file:
            self.data = json.load(json_file)

        print("Opening files...")
        
        return self.clean() # Dictionary of urls -> {title, content, links}


    def clean_results(self, links):
        pages = {page: self.data[page]["title"] for page in links}

        print("Cleaned search results...")
        
        return pages
    
    
    def clean(self):
        pages = dict()

        for page in self.data:
            pages[page["url"]] = { "title": page["title"],
                                   "content": page["content"],
                                   "links": set(page["links"]) - {page["url"]},
                                }
            
        # Only include links to other pages in the corpus
        for filename in pages:
            pages[filename]["links"] = set(
                link for link in pages[filename]["links"]
                if link in pages
            )
            
        print("Cleaned data...")

        return pages


def main():
    index = Index()

    #index.create_indexes()
    index.load_data()
    print(index.search(query="Python language"))


#main()
#if __name__ == "__main__":
#    main()