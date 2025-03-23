import os
import re
import json
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Dict, List, Tuple
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import math
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
import base64

# Инициализация NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class MiniSearchEngine:
    def __init__(self, storage_path: str = "search_data"):
        self.storage_path = storage_path
        self.index_file = os.path.join(storage_path, "inverted_index.json")
        self.docs_info_file = os.path.join(storage_path, "docs_info.json")
        self.stemmer = SnowballStemmer("russian")
        self.stop_words = set(stopwords.words('russian'))
        self.inverted_index = {}
        self.documents = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        
        os.makedirs(storage_path, exist_ok=True)

    def crawl(self, start_urls: List[str], max_depth: int = 2, max_pages: int = 100):
        """Простой краулер для сбора страниц"""
        visited = set()
        to_visit = [(url, 0) for url in start_urls]
        doc_id = 0
        
        while to_visit and len(visited) < max_pages:
            url, depth = to_visit.pop(0)
            if url in visited or depth > max_depth:
                continue
                
            try:
                response = requests.get(url, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Извлечение текста
                text = soup.get_text()
                text = re.sub(r'\s+', ' ', text.lower())
                text = re.sub(r'[^\w\s]', '', text)
                
                self.documents[f"doc_{doc_id}"] = {
                    'content': text,
                    'url': url,
                    'title': soup.title.string if soup.title else url
                }
                doc_id += 1
                visited.add(url)
                
                # Поиск ссылок для дальнейшего обхода
                if depth < max_depth:
                    for link in soup.find_all('a', href=True):
                        next_url = link['href']
                        if next_url.startswith('http'):
                            to_visit.append((next_url, depth + 1))
                            print(next_url)
                            
            except Exception as e:
                print(f"Ошибка при загрузке {url}: {e}")
                
        self._save_docs_info()

    def _save_docs_info(self):
        """Сохранение информации о документах"""
        with open(self.docs_info_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def tokenize_and_stem(self, text: str) -> List[str]:
        """Токенизация и стемминг"""
        tokens = re.findall(r'\b\w+\b', text)
        return [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]

    def build_index(self):
        """Построение инвертированного индекса"""
        inverted_index = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "positions": []}))
        total_length = 0
        
        for doc_id, doc_info in self.documents.items():
            tokens = self.tokenize_and_stem(doc_info['content'])
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            for pos, token in enumerate(tokens):
                inverted_index[token][doc_id]["frequency"] += 1
                inverted_index[token][doc_id]["positions"].append(pos)
        
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 1
        self.inverted_index = dict(inverted_index)
        
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, ensure_ascii=False)

    def calculate_bm25(self, term: str, doc_id: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Расчет BM25"""
        if term not in self.inverted_index or doc_id not in self.inverted_index[term]:
            return 0
            
        tf = self.inverted_index[term][doc_id]["frequency"]
        doc_len = self.doc_lengths[doc_id]
        idf = math.log((len(self.documents) - len(self.inverted_index[term]) + 0.5) / 
                      (len(self.inverted_index[term]) + 0.5) + 1)
        
        return idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self.avg_doc_length)))

    def simple_pagerank(self) -> Dict[str, float]:
        """Простая реализация PageRank (на основе количества упоминаний URL)"""
        ranks = defaultdict(float)
        for doc_id in self.documents:
            ranks[doc_id] = 1.0
        
        for _ in range(5):  # 5 итераций
            new_ranks = defaultdict(float)
            for doc_id, doc_info in self.documents.items():
                content = doc_info['content']
                for other_id, other_info in self.documents.items():
                    if other_info['url'] in content:
                        new_ranks[other_id] += ranks[doc_id] / sum(1 for d in self.documents.values() 
                                                                 if other_info['url'] in d['content'])
            ranks = new_ranks
        return dict(ranks)

    def search(self, query: str) -> List[Tuple[str, float]]:
        """Поиск с ранжированием"""
        query_tokens = self.tokenize_and_stem(query)
        if not query_tokens:
            return []
            
        scores = defaultdict(float)
        pagerank = self.simple_pagerank()
        
        # Расчет комбинированного рейтинга (BM25 + PageRank)
        for token in query_tokens:
            if token in self.inverted_index:
                for doc_id in self.inverted_index[token]:
                    bm25_score = self.calculate_bm25(token, doc_id)
                    pr_score = pagerank.get(doc_id, 1.0)
                    scores[doc_id] += bm25_score * 0.7 + pr_score * 0.3
        
        # Сортировка по убыванию рейтинга
        results = [(doc_id, score) for doc_id, score in scores.items()]
        return sorted(results, key=lambda x: x[1], reverse=True)

    def visualize_results(self, results: List[Tuple[str, float]]) -> str:
        """Визуализация результатов"""
        if not results:
            return ""
            
        docs, scores = zip(*results[:10])  # Топ-10 результатов
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(docs)), scores)
        plt.xticks(range(len(docs)), [self.documents[d]['title'][:20] for d in docs], rotation=45)
        plt.xlabel('Документы')
        plt.ylabel('Рейтинг релевантности')
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

# Веб-интерфейс
app = Flask(__name__)
search_engine = MiniSearchEngine()

@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    graph = ""
    if request.method == 'POST':
        query = request.form['query']
        results = search_engine.search(query)
        graph = search_engine.visualize_results(results)
    
    return render_template('search.html', results=results, graph=graph, documents=search_engine.documents)

def main():
    # Инициализация и запуск
    start_urls = [
        "https://ru.wikipedia.org/wiki/Python",
        "https://ru.wikipedia.org/wiki/Поисковая_система"
    ]
    
    print("Сбор данных...")
    search_engine.crawl(start_urls)
    print("Построение индекса...")
    search_engine.build_index()
    print("Запуск веб-сервера...")
    app.run(debug=True)

if __name__ == "__main__":
    main()