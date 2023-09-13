import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import math
import string
#nltk.download('stopwords')

N = 5


def main():
    # Open file
    with open('wiki.json', 'r') as json_file:
        data = json.load(json_file)

    files = clean(data) # Clean data to {title: content}
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    print("Tokenized file words...")

    file_idfs = compute_idfs(file_words) # Compute file idfs

    query = set(tokenize(input("Query: ")))

    filenames = top_files(query, file_words, file_idfs, N)

    print(filenames)



def clean(data):
    pages = dict()

    for page in data[:150]:
        pages[page["url"]] = page["content"]
    print("Cleaned data...")

    return pages


def tokenize(document):
    tokens = nltk.word_tokenize(document.lower())

    words = [
        token for token in tokens
        if token not in stopwords.words('english') and # Remove determinants etc.
        token not in string.punctuation # Remove punctuation
    ]

    return words


def compute_idfs(documents):
    idfs = dict()
    all_words = set(word for words in documents.values() for word in words) # All words in documents without repeating

    for word in all_words:
        freq = sum(1 for words in documents.values() if word in words)
        idf = math.log(len(documents) / freq)
        idfs[word] = idf

    print("Computed idfs...")
    return idfs


def top_files(query, files, idfs, n):
    scores = dict()

    # Iterate over files assigning score to each file
    for file, words in files.items():
        file_score = 0

        for word in query: # Match query words with file words
            if word in words:
                tf = words.count(word) / len(words) # Term frequency
                tf_idf_score = tf * idfs[word]
                file_score += tf_idf_score # Add to file score
        scores[file] = file_score # File score
    print("Searched top files...")
    # Sort/filter files by scores, return to n files
    return sorted(list(files.keys()), key=lambda x: -scores[x])[:n]


def top_sentences(query, sentences, idfs, n):
    sentences_scores = dict()
    


main()