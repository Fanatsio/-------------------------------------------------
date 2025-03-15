import os
import re
import json
import math
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download("stopwords")

# Генерация документов
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

# Предобработка текста
def read_and_preprocess_documents(folder="./Lab4/docs"):
    documents = {}
    russian_stopwords = set(stopwords.words("russian"))
    stemmer = SnowballStemmer("russian")
    
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
                text = re.sub(r'[^\w\s]', '', text).lower()
                tokens = re.findall(r'\b\w+\b', text)
                tokens = [stemmer.stem(token) for token in tokens if token not in russian_stopwords]
                documents[filename] = tokens
    
    return documents

# Вычисление TF (term frequency)
def compute_tf(documents):
    tf = {}
    for doc, words in documents.items():
        word_count = Counter(words)
        total_words = len(words)
        tf[doc] = {word: count / total_words for word, count in word_count.items()}
    return tf

# Вычисление IDF (inverse document frequency)
def compute_idf(documents):
    idf = {}
    total_docs = len(documents)
    all_words = set(word for words in documents.values() for word in words)
    
    for word in all_words:
        containing_docs = sum(1 for words in documents.values() if word in words)
        idf[word] = math.log(total_docs / (1 + containing_docs)) + 1
    
    return idf

# Вычисление TF-IDF
def compute_tfidf(tf, idf):
    tfidf = {}
    for doc, words in tf.items():
        tfidf[doc] = {word: tf_val * idf[word] for word, tf_val in words.items()}
    return tfidf

# Поиск и ранжирование документов
def search_query(query, tfidf, documents):
    stemmer = SnowballStemmer("russian")
    query_tokens = re.findall(r'\b\w+\b', query.lower())
    query_tokens = [stemmer.stem(token) for token in query_tokens]
    
    query_vector = {word: 1 for word in query_tokens}
    
    scores = {}
    for doc, words in tfidf.items():
        score = sum(words.get(word, 0) * query_vector.get(word, 0) for word in query_tokens)
        scores[doc] = score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Оценка качества поиска
def evaluate_search(results, relevant_docs):
    retrieved_docs = set(doc for doc, _ in results)
    relevant_docs = set(relevant_docs)
    
    true_positives = len(retrieved_docs & relevant_docs)
    precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
    recall = true_positives / len(relevant_docs) if relevant_docs else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall else 0
    
    return precision, recall, f1_score

# Основной процесс
generate_documents()
documents = read_and_preprocess_documents()
tf = compute_tf(documents)
idf = compute_idf(documents)
tfidf = compute_tfidf(tf, idf)

# Пример запроса
query = "машинное обучение"
results = search_query(query, tfidf, documents)
print("Ранжированные результаты:")
print(results)