import os
import re
import json
import numpy as np
from collections import defaultdict
from nltk.stem import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity

# Функция генерации документов
def generate_documents(folder="./Lab4/docs"):
    os.makedirs(folder, exist_ok=True)
    if any(os.scandir(folder)):
        return
    
    texts = {
        "doc1.txt": "Машинное обучение развивается быстро. Оно позволяет компьютерам учиться на данных. Многие отрасли его используют. Алгоритмы анализируют огромные массивы информации. Это открывает новые возможности в науке и бизнесе.",
        "doc2.txt": "Обработка естественного языка – ключевая область ИИ. Она помогает машинам понимать речь и текст. Это применяется в чат-ботах и переводчиках. Современные модели способны понимать контекст. Это делает взаимодействие с машинами удобнее.",
        "doc3.txt": "Индексация текста улучшает производительность поисковых систем. Она делает поиск информации эффективным. Без индексации системы работали бы медленнее. Современные алгоритмы оптимизируют хранение данных. Это снижает нагрузку на серверы.",
        "doc4.txt": "Глубокое обучение использует нейросети. Оно достигло прорывов в областях, включая распознавание изображений. Такие технологии помогают в медицине. Диагностика заболеваний становится точнее. Это спасает жизни.",
        "doc5.txt": "Наука о данных – это междисциплинарная область. Она объединяет статистику, программирование и знания о предметной области. Анализ данных помогает принимать решения. Компании используют его для прогнозирования трендов. Это улучшает бизнес-процессы.",
        "doc6.txt": "Искусственный интеллект влияет на повседневную жизнь. Он используется в голосовых помощниках и рекомендательных системах. AI помогает автоматизировать рутинные задачи. Это увеличивает производительность труда. Также AI применяется в медицине и финансах.",
        "doc7.txt": "Компьютерное зрение позволяет машинам интерпретировать изображения. Оно используется в медицине и системах безопасности. Камеры с AI могут анализировать дорожную ситуацию. Это помогает предотвращать аварии. Также технологии применяются в промышленности.",
        "doc8.txt": "Аналитика больших данных помогает принимать решения. Компании используют её для прогнозов и анализа тенденций. Современные системы анализируют поведение пользователей. Это позволяет улучшать сервисы и персонализировать предложения. Бизнес получает конкурентное преимущество.",
        "doc9.txt": "Кибербезопасность крайне важна в цифровую эпоху. Шифрование и аутентификация защищают конфиденциальные данные. Хакеры постоянно совершенствуют атаки. Поэтому требуется разработка новых методов защиты. Компании инвестируют в безопасность своих пользователей.",
        "doc10.txt": "Облачные вычисления обеспечивают масштабируемую инфраструктуру. Бизнес использует их для хранения данных и обработки информации. Облака позволяют экономить ресурсы. Они обеспечивают гибкость и надежность. Компании переходят на облачные технологии для повышения эффективности."
    }
    for filename, text in texts.items():
        with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
            f.write(text)

def read_and_preprocess_documents(folder="./Lab4/docs"):
    stop_words = set("и в на с под для это его её их".split())  # Добавь полный список стоп-слов
    documents = {}
    stemmer = SnowballStemmer("russian")
    
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = re.sub(r'[^\w\s]', '', f.read().lower())
                tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
                documents[filename] = tokens
    
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
    
    for doc, words in term_freq.items():
        for word in vocab:
            tf = words[word] / len(documents[doc]) if word in words else 0
            idf = np.log((num_docs + 1) / (1 + doc_freq[word])) + 1
            tf_idf[doc][word] = tf * idf
    
    return tf_idf, vocab

def query_tfidf(query, tf_idf, vocab):
    stemmer = SnowballStemmer("russian")
    query_tokens = [stemmer.stem(word) for word in query.lower().split()]
    
    # Формируем TF-IDF вектор запроса
    query_tf = defaultdict(int)
    for word in query_tokens:
        if word in vocab:
            query_tf[word] += 1
    
    num_docs = len(tf_idf)
    query_vec = np.array([query_tf[word] / len(query_tokens) * (np.log((num_docs + 1) / (1 + sum(1 for doc in tf_idf if word in tf_idf[doc]))) + 1) if word in vocab else 0 for word in vocab])
    
    scores = {}
    for doc, word_scores in tf_idf.items():
        doc_vec = np.array([word_scores.get(word, 0) for word in vocab])
        
        if np.linalg.norm(doc_vec) == 0 or np.linalg.norm(query_vec) == 0:
            scores[doc] = 0
        else:
            scores[doc] = cosine_similarity([query_vec], [doc_vec])[0][0]
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True), query_vec

# Запуск кода
generate_documents()
documents = read_and_preprocess_documents()
tf_idf, vocab = compute_tf_idf(documents)
query = "машинное обучение и анализ данных"
results, query_vector = query_tfidf(query, tf_idf, vocab)
print(vocab)
print("TF-IDF вектор для запроса:", query_vector.tolist())  
print("Ранжированные документы:", [(doc, round(float(score), 4)) for doc, score in results])
