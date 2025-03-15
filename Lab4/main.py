import os
import re
import json
from collections import defaultdict
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')

def read_and_preprocess_documents(folder="docs"):
    documents = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
                text = re.sub(r'[^\w\s]', '', text).lower()
                documents[filename] = text
    return documents

def tokenize_and_stem(text):
    tokens = re.findall(r'\b\w+\b', text)
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return filtered_tokens

def build_inverted_index(documents, index_file="inverted_index.json"):
    inverted_index = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "positions": []}))
    
    for doc_id, text in documents.items():
        tokens = tokenize_and_stem(text)
        for index, token in enumerate(tokens):
            inverted_index[token][doc_id]["frequency"] += 1
            inverted_index[token][doc_id]["positions"].append(index)
    
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=4)
    
    return inverted_index

def search(query, inverted_index):
    query_tokens = tokenize_and_stem(query)
    result_docs = defaultdict(list)
    
    for token in query_tokens:
        if token in inverted_index:
            for doc_id, data in inverted_index[token].items():
                result_docs[doc_id].append((token, data["frequency"], data["positions"]))
    
    return dict(result_docs)

def search_phrase(phrase, inverted_index):
    phrase_tokens = tokenize_and_stem(phrase)
    if not phrase_tokens:
        return []
    
    first_word, *rest_words = phrase_tokens
    if first_word not in inverted_index:
        return []
    
    candidate_docs = set(inverted_index[first_word].keys())
    for word in rest_words:
        if word not in inverted_index:
            return []
        candidate_docs &= set(inverted_index[word].keys())
    
    matching_docs = []
    for doc_id in candidate_docs:
        positions = [set(inverted_index[word][doc_id]["positions"]) for word in phrase_tokens]
        for pos in positions[0]:
            if all(pos + i in positions[i] for i in range(len(positions))):
                matching_docs.append((doc_id, pos))
                break
    
    return matching_docs

documents = read_and_preprocess_documents()

inverted_index = build_inverted_index(documents)

query = input("Введите фразу для поиска --->\t")
print("Поиск отдельных слов:", search(query, inverted_index))
print("Поиск фразы:", search_phrase(query, inverted_index))
