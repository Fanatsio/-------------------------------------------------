import os
import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import time

class SearchEngine:
    def __init__(self, folder: str = "Lab4/Docs", index_file: str = "Lab4/inverted_index.json"):
        """Инициализация поискового движка."""
        self.folder = folder
        self.index_file = index_file
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words('english'))
        self.inverted_index = None
        self.documents = None
        
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

    def read_documents(self) -> Dict[str, str]:
        """Чтение и предобработка документов из указанной папки."""
        documents = {}
        try:
            for filename in os.listdir(self.folder):
                if filename.endswith(".txt"):
                    filepath = os.path.join(self.folder, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read().lower()
                        text = re.sub(r'[^\w\s]', '', text)
                        documents[filename] = text
        except FileNotFoundError:
            print(f"Папка {self.folder} не найдена")
            return {}
        except Exception as e:
            print(f"Ошибка при чтении файлов: {e}")
            return {}
        return documents

    def tokenize_and_stem(self, text: str) -> List[str]:
        """Токенизация и стемминг текста."""
        tokens = re.findall(r'\b\w+\b', text)
        return [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]

    def build_inverted_index(self, documents: Dict[str, str]) -> Dict:
        """Построение инвертированного индекса."""
        inverted_index = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "positions": []}))
        
        for doc_id, text in documents.items():
            tokens = self.tokenize_and_stem(text)
            for pos, token in enumerate(tokens):
                inverted_index[token][doc_id]["frequency"] += 1
                inverted_index[token][doc_id]["positions"].append(pos)
        
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(dict(inverted_index), f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Ошибка при сохранении индекса: {e}")
        
        self.inverted_index = inverted_index
        return inverted_index

    def search_words(self, query: str) -> Dict[str, List[Tuple[str, int, List[int]]]]:
        """Поиск отдельных слов с использованием инвертированного индекса."""
        if not self.inverted_index:
            return {}
            
        query_tokens = self.tokenize_and_stem(query)
        result_docs = defaultdict(list)
        
        for token in query_tokens:
            if token in self.inverted_index:
                for doc_id, data in self.inverted_index[token].items():
                    result_docs[doc_id].append((token, data["frequency"], data["positions"]))
        
        return dict(result_docs)

    def brute_force_search_words(self, query: str) -> Dict[str, List[Tuple[str, int, List[int]]]]:
        """Поиск отдельных слов методом перебора."""
        if not self.documents:
            return {}
            
        query_tokens = self.tokenize_and_stem(query)
        result_docs = defaultdict(list)
        
        for doc_id, text in self.documents.items():
            doc_tokens = self.tokenize_and_stem(text)
            for token in query_tokens:
                frequency = 0
                positions = []
                for pos, doc_token in enumerate(doc_tokens):
                    if doc_token == token:
                        frequency += 1
                        positions.append(pos)
                if frequency > 0:
                    result_docs[doc_id].append((token, frequency, positions))
        
        return dict(result_docs)

    def search_phrase(self, phrase: str) -> List[Tuple[str, int]]:
        """Поиск точной фразы с использованием инвертированного индекса."""
        if not self.inverted_index:
            return []
            
        phrase_tokens = self.tokenize_and_stem(phrase)
        if not phrase_tokens:
            return []
            
        first_word, *rest_words = phrase_tokens
        if first_word not in self.inverted_index:
            return []
            
        candidate_docs = set(self.inverted_index[first_word].keys())
        for word in rest_words:
            if word not in self.inverted_index:
                return []
            candidate_docs &= set(self.inverted_index[word].keys())
        
        matching_docs = []
        for doc_id in candidate_docs:
            positions = [self.inverted_index[word][doc_id]["positions"] for word in phrase_tokens]
            for pos in positions[0]:
                if all(pos + i in positions[i] for i in range(len(positions))):
                    matching_docs.append((doc_id, pos))
                    break
        
        return matching_docs

    def brute_force_search(self, phrase: str) -> List[Tuple[str, int]]:
        """Поиск точной фразы методом перебора."""
        if not self.documents:
            return []
            
        phrase_tokens = self.tokenize_and_stem(phrase)
        if not phrase_tokens:
            return []
            
        matching_docs = []
        for doc_id, text in self.documents.items():
            doc_tokens = self.tokenize_and_stem(text)
            for i in range(len(doc_tokens) - len(phrase_tokens) + 1):
                if doc_tokens[i:i + len(phrase_tokens)] == phrase_tokens:
                    matching_docs.append((doc_id, i))
                    break
        
        return matching_docs

def main():
    """Основная функция для запуска поискового движка."""
    search_engine = SearchEngine()
    
    # Подготовка данных
    search_engine.documents = search_engine.read_documents()
    if not search_engine.documents:
        print("Нет документов для обработки")
        return
        
    inverted_index = search_engine.build_inverted_index(search_engine.documents)
    
    # Поисковый цикл
    while True:
        query = input("Введите фразу для поиска (или 'exit' для выхода) --->\t")
        if query.lower() == 'exit':
            break
            
        # Поиск отдельных слов с инвертированным индексом
        start_time = time.time()
        word_index_results = search_engine.search_words(query)
        word_index_time = time.time() - start_time
        
        # Поиск отдельных слов перебором
        start_time = time.time()
        word_brute_results = search_engine.brute_force_search_words(query)
        word_brute_time = time.time() - start_time
        
        # Поиск фразы с инвертированным индексом
        start_time = time.time()
        phrase_index_results = search_engine.search_phrase(query)
        phrase_index_time = time.time() - start_time
        
        # Поиск фразы перебором
        start_time = time.time()
        phrase_brute_results = search_engine.brute_force_search(query)
        phrase_brute_time = time.time() - start_time
        
        # Вывод результатов
        print("\nРезультаты поиска отдельных слов с инвертированным индексом:")
        for doc, tokens in word_index_results.items():
            print(f"{doc}: {tokens}")
        print(f"Время выполнения: {word_index_time:.6f} секунд")
        
        print("\nРезультаты поиска отдельных слов перебором:")
        for doc, tokens in word_brute_results.items():
            print(f"{doc}: {tokens}")
        print(f"Время выполнения: {word_brute_time:.6f} секунд")
        
        print("\nРезультаты поиска фразы с инвертированным индексом:")
        for doc, pos in phrase_index_results:
            print(f"{doc}: найдено на позиции {pos}")
        print(f"Время выполнения: {phrase_index_time:.6f} секунд")
        
        print("\nРезультаты поиска фразы перебором:")
        for doc, pos in phrase_brute_results:
            print(f"{doc}: найдено на позиции {pos}")
        print(f"Время выполнения: {phrase_brute_time:.6f} секунд")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()