import heapq
import time
import random
import matplotlib.pyplot as plt
import networkx as nx

def dfs(graph, start, target=None):
    visited = set()
    order = []
    stack = [start]

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            if target is not None and vertex == target:
                break
            stack.extend(sorted(set(graph[vertex].keys()) - visited, reverse=True))

    return order

def bfs(graph, start, target=None):
    visited = set()
    queue = [start]
    order = []

    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            if target is not None and vertex == target:
                break
            queue.extend(sorted(set(graph[vertex].keys()) - visited))

    return order

def dijkstra(graph, start, target=None):
    visited = set()
    pq = [(0, start)]
    distances = {start: 0}
    previous_vertices = {start: None}
    order = []

    while pq:
        cost, vertex = heapq.heappop(pq)
        if vertex in visited:
            continue
        visited.add(vertex)
        order.append(vertex)
        if target is not None and vertex == target:
            break
        for neighbor, weight in graph.get(vertex, {}).items():
            if neighbor not in visited:
                new_cost = cost + weight
                if neighbor not in distances or new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    previous_vertices[neighbor] = vertex
                    heapq.heappush(pq, (new_cost, neighbor))

    path = []
    distance = float('inf')
    if target is not None and target in distances:
        current_vertex = target
        while current_vertex is not None:
            path.append(current_vertex)
            current_vertex = previous_vertices.get(current_vertex)
        path = path[::-1]
        distance = distances.get(target, float('inf'))
        if distance == float('inf'):
            path = []

    return order, distance, path

def visualize_graph(graph, title="Граф"):
    G = nx.Graph()
    for vertex in graph:
        for neighbor, weight in graph[vertex].items():
            G.add_edge(vertex, neighbor, weight=weight)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.show()

def measure_time(algorithm, graph, start, target=None):
    start_time = time.perf_counter()
    result = algorithm(graph, start, target)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, result

def process_graph(graph, name):
    print(f"\n{name}:")
    for algorithm, label in [(bfs, "BFS"), (dfs, "DFS"), (dijkstra, "Дейкстра")]:
        if label == "Дейкстра":
            time_taken, result = measure_time(algorithm, graph, 0, 4)
            order, distance, path = result
            print(f"{label}: {time_taken:.6f} сек, Порядок обхода: {order}, Длина пути: {distance}, Путь: {path}")
        else:
            time_taken, result = measure_time(algorithm, graph, 0)
            print(f"{label}: {time_taken:.6f} сек, Порядок обхода: {result}")
        visualize_graph(graph, title=name)

def generate_random_graph(size):
    return {i: {random.choice(range(size)): random.randint(1, 10) for _ in range(random.randint(1, min(5, size-1)))} for i in range(size)}

def plot_execution_times(sizes, bfs_times, dfs_times, dijkstra_times):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, bfs_times, marker='o', label='BFS', linestyle='-', color='r')
    plt.plot(sizes, dfs_times, marker='o', label='DFS', linestyle='-', color='g')
    plt.plot(sizes, dijkstra_times, marker='o', label='Дейкстра', linestyle='-', color='b')
    plt.xlabel('Размер графа (количество вершин)')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Сравнение времени выполнения BFS, DFS и Дейкстры')
    plt.legend()
    plt.grid(True)
    plt.show()

# Графы
small_graph = {
    0: {1: 1, 2: 1},
    1: {0: 1, 3: 1, 4: 1},
    2: {0: 1, 4: 1},
    3: {1: 1, 4: 1},
    4: {1: 1, 2: 1, 3: 1}
}

medium_graph = {
    0: {1: 1, 2: 1, 3: 1},
    1: {0: 1, 4: 1, 5: 1},
    2: {0: 1, 6: 1, 7: 1},
    3: {0: 1, 8: 1, 9: 1},
    4: {1: 1, 10: 1, 11: 1},
    5: {1: 1, 12: 1, 13: 1},
    6: {2: 1, 14: 1, 15: 1},
    7: {2: 1, 16: 1, 17: 1},
    8: {3: 1, 18: 1, 19: 1},
    9: {3: 1, 10: 1, 11: 1},
    10: {4: 1, 9: 1, 12: 1},
    11: {4: 1, 9: 1, 13: 1},
    12: {5: 1, 10: 1, 14: 1},
    13: {5: 1, 11: 1, 15: 1},
    14: {6: 1, 12: 1, 16: 1},
    15: {6: 1, 13: 1, 17: 1},
    16: {7: 1, 14: 1, 18: 1},
    17: {7: 1, 15: 1, 19: 1},
    18: {8: 1, 16: 1, 19: 1},
    19: {8: 1, 17: 1, 18: 1}
}

weighted_graph = {
    0: {1: 4, 2: 1},
    1: {0: 4, 3: 1},
    2: {0: 1, 3: 2},
    3: {1: 1, 2: 2, 4: 3},
    4: {3: 3}
}

# Обработка графов
for graph, name in [(small_graph, "Малый граф"), (medium_graph, "Средний граф"), (weighted_graph, "Взвешенный граф")]:
    process_graph(graph, name)

# Случайные графы
sizes = [10, 20, 40, 60, 80, 100]
bfs_times = []
dfs_times = []
dijkstra_times = []

for size in sizes:
    random_graph = generate_random_graph(size)
    print(f"\nСлучайный граф из {size} вершин")
    for algorithm, times_list, label in [(bfs, bfs_times, "BFS"), (dfs, dfs_times, "DFS"), (dijkstra, dijkstra_times, "Дейкстра")]:
        time_taken, _ = measure_time(algorithm, random_graph, 0)
        times_list.append(time_taken)
        print(f"{label}: {time_taken:.6f} сек")

plot_execution_times(sizes, bfs_times, dfs_times, dijkstra_times)
