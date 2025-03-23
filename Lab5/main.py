import os
import re
import numpy as np
from collections import defaultdict
from nltk.stem import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import time

def read_and_preprocess_documents(folder="Lab4/Docs"):
    documents = {}
    stop_words = set(stopwords.words("russian"))
    stemmer = SnowballStemmer("russian")
    
    print("\n=== Загрузка и предобработка документов ===")
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = re.sub(r'[^\w\s]', '', f.read().lower())
                tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
                documents[filename] = tokens
                print(f"Документ {filename}: {tokens[:10]}... (всего {len(tokens)} слов)")
    return documents

def compute_tf_idf(documents):
    term_freq = defaultdict(lambda: defaultdict(int))
    doc_freq = defaultdict(int)
    
    for doc, words in documents.items():
        for word in words:
            term_freq[doc][word] += 1
        for word in set(words):
            doc_freq[word] += 1
    
    num_docs = len(documents)
    tf_idf = defaultdict(lambda: defaultdict(float))
    vocab = set(doc_freq.keys())
    
    print(f"Словарь (vocab): {list(vocab)[:10]}... (всего {len(vocab)} терминов)")
    for doc, words in term_freq.items():
        for word in vocab:
            tf = words[word] / len(documents[doc]) if word in words else 0
            idf = np.log((num_docs + 1) / (1 + doc_freq[word])) + 1
            tf_idf[doc][word] = tf * idf
    return tf_idf, vocab

def query_tfidf(query, tf_idf, vocab):
    stemmer = SnowballStemmer("russian")
    query_tokens = [stemmer.stem(word) for word in query.lower().split()]
    print(f"Токены запроса: {query_tokens}")
    
    query_tf = defaultdict(int)
    for word in query_tokens:
        if word in vocab:
            query_tf[word] += 1
    
    num_docs = len(tf_idf)
    query_vec = np.array([query_tf[word] / len(query_tokens) * (np.log((num_docs + 1) / (1 + sum(1 for doc in tf_idf if word in tf_idf[doc]))) + 1) if word in vocab else 0 for word in vocab])
    print(f"Длина вектора запроса: {len(query_vec)}, норма: {np.linalg.norm(query_vec):.4f}")
    
    scores = {}
    for doc, word_scores in tf_idf.items():
        doc_vec = np.array([word_scores.get(word, 0) for word in vocab])
        if np.linalg.norm(doc_vec) == 0 or np.linalg.norm(query_vec) == 0:
            scores[doc] = 0
        else:
            scores[doc] = cosine_similarity([query_vec], [doc_vec])[0][0]
    return sorted(scores.items(), key=lambda x: x[1], reverse=True), query_vec

def simple_word_count(query, documents):
    stemmer = SnowballStemmer("russian")
    query_tokens = set(stemmer.stem(word) for word in query.lower().split())
    scores = {}
    
    for doc, words in documents.items():
        count = sum(1 for word in words if word in query_tokens)
        scores[doc] = count / len(words) if len(words) > 0 else 0
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def evaluate_search(results, relevant_docs, top_n=3):
    retrieved = [doc for doc, _ in results[:top_n]]
    print(f"Топ-{top_n} документов: {retrieved}")
    true_positives = len(set(retrieved) & set(relevant_docs))
    precision = true_positives / top_n if top_n > 0 else 0
    recall = true_positives / len(relevant_docs) if len(relevant_docs) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

if __name__ == "__main__":
    documents = read_and_preprocess_documents()
    tf_idf, vocab = compute_tf_idf(documents)
    
    queries = ["эльфийский хлеб", "фродо кольцо", "гэндальф битва"]
    
    for query in queries:
        print(f"\n=== Обработка запроса '{query}' ===")

        # Релевантные документы
        relevant_docs = {
            "эльфийский хлеб": ["Властелин_колец_Возвращение_короля_часть_6.txt", "Властелин_колец_Возвращение_короля_часть_7.txt"],
            "фродо кольцо": ["Властелин_колец_Братство_кольца_часть_1.txt", "Властелин_колец_Братство_кольца_часть_2.txt"],
            "гэндальф битва": ["Властелин_колец_Возвращение_короля_часть_1.txt", "Властелин_колец_Возвращение_короля_часть_4.txt"]
        }.get(query.lower())
        print(f"\nРелевантные документы для запроса '{query}': {relevant_docs}")
        
        # TF-IDF
        start_time_tfidf = time.perf_counter()
        tfidf_results, query_vector = query_tfidf(query, tf_idf, vocab)
        end_time_tfidf = time.perf_counter()
        tfidf_time = end_time_tfidf - start_time_tfidf
        
        print("\nРанжированные документы (TF-IDF):")
        for doc, score in tfidf_results:
            print(f"{doc}: {round(float(score), 4)}")
        precision_tfidf, recall_tfidf, f1_tfidf = evaluate_search(tfidf_results, relevant_docs)
        
        # Простой подсчёт
        start_time_simple = time.perf_counter()
        simple_results = simple_word_count(query, documents)
        end_time_simple = time.perf_counter()
        simple_time = end_time_simple - start_time_simple
        
        print("\nРанжированные документы (простой подсчёт):")
        for doc, score in simple_results:
            print(f"{doc}: {round(float(score), 4)}")
        precision_simple, recall_simple, f1_simple = evaluate_search(simple_results, relevant_docs)
        
        # Метрики качества
        print("\nМетрики качества (TF-IDF):")
        print(f"Precision: {precision_tfidf:.4f}, Recall: {recall_tfidf:.4f}, F1-score: {f1_tfidf:.4f}")
        
        print("\nМетрики качества (простой подсчёт):")
        print(f"Precision: {precision_simple:.4f}, Recall: {recall_simple:.4f}, F1-score: {f1_simple:.4f}")
        
        # Сравнение времени
        print("\nСравнение времени выполнения:")
        print(f"TF-IDF: {tfidf_time:.4f} секунд")
        print(f"Простой подсчёт: {simple_time:.4f} секунд")
        print(f"Разница: {abs(tfidf_time - simple_time):.4f} секунд (TF-IDF {'быстрее' if tfidf_time < simple_time else 'медленнее'})")