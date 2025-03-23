import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PageRankResult:
    """Класс для хранения результатов PageRank"""
    values: np.ndarray
    history: np.ndarray
    iterations: int

class PageRankAnalyzer:
    """Класс для анализа и визуализации PageRank"""
    def __init__(self):
        self.graph = self._create_graph()
        self.node_labels = [chr(65 + i) for i in range(5)]  # A, B, C, D, E
        
    @staticmethod
    def _create_graph() -> np.ndarray:
        """Создание графа ссылок"""
        return np.array([
            [0, 1, 1, 0, 0],  # A → B, C
            [1, 0, 0, 1, 0],  # B → A, D
            [0, 0, 0, 1, 1],  # C → D, E
            [0, 0, 0, 0, 1],  # D → E
            [0, 0, 0, 0, 0]   # E → -
        ])

    def calculate_pagerank(self, alpha: float, max_iter: int = 100, tol: float = 1e-6) -> PageRankResult:
        """Вычисление PageRank"""
        n = self.graph.shape[0]
        pr = np.ones(n) / n
        M = np.nan_to_num(self.graph / self.graph.sum(axis=0, keepdims=True))
        history = [pr.copy()]
        
        for _ in range(max_iter):
            pr_new = (1 - alpha) / n + alpha * M @ pr
            if np.all(np.abs(pr_new - pr) < tol):
                break
            pr = pr_new
            history.append(pr.copy())
            
        return PageRankResult(pr, np.array(history), len(history))

    def draw_graph(self) -> None:
        """Визуализация структуры графа"""
        plt.figure(figsize=(8, 6))
        G = nx.DiGraph(self.graph)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, arrowsize=20, font_size=12, font_weight='bold')
        plt.title("Структура графа ссылок", fontsize=14, pad=10)
        plt.show()

    def plot_history(self, result: PageRankResult, alpha: float) -> None:
        """Визуализация истории PageRank"""
        plt.figure(figsize=(12, 7))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
        
        for i, label in enumerate(self.node_labels):
            plt.plot(result.history[:, i], label=f'Страница {label}',
                    color=colors[i], linewidth=2.5, marker='o', markersize=5)
            
        plt.xlabel('Итерации', fontsize=12)
        plt.ylabel('Значение PageRank', fontsize=12)
        plt.title(f'Динамика PageRank (α={alpha}, {result.iterations} итераций)',
                 fontsize=14, pad=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def print_results(self, results: Dict[float, np.ndarray], iterations: Dict[float, int]) -> None:
        """Вывод таблицы результатов"""
        alphas = sorted(results.keys())
        print("\nРЕЗУЛЬТАТЫ PageRank")
        print("Страница | " + " | ".join(f"α={a:<5}" for a in alphas))
        print("-" * (12 + 10 * len(alphas)))
        
        for i, label in enumerate(self.node_labels):
            print(f"{label:<8} | " + " | ".join(
                f"{results[a][i]:.4f}" for a in alphas))

    @staticmethod
    def print_analysis() -> None:
        """Вывод анализа результатов"""
        print("\nАНАЛИЗ РЕЗУЛЬТАТОВ")
        print("1. Влияние структуры:")
        print("   - Страницы с большим числом входящих ссылок имеют высокий PR")
        print("   - Страницы без исходящих ссылок аккумулируют значимость")
        print("2. Влияние α:")
        print("   - Высокий α усиливает влияние структуры графа")
        print("   - Низкий α делает распределение более равномерным")

    def run_analysis(self, alphas: List[float] = [0.6, 0.7, 0.85, 0.95]) -> None:
        self.draw_graph()
        
        results = {}
        iterations = {}
        for alpha in alphas:
            result = self.calculate_pagerank(alpha)
            results[alpha] = result.values
            iterations[alpha] = result.iterations
            self.plot_history(result, alpha)
            
        self.print_results(results, iterations)
        self.print_analysis()

def main():
    analyzer = PageRankAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()