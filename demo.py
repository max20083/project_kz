"""
Демонстрация работы K-means на синтетических 2D-данных.
"""

from __future__ import annotations

import math
import random

from k_means import assign_clusters, inertia, k_means , del_temp_files


def normal_sample() -> float:
    """Одно значение из стандартного нормального распределения (Box-Muller)."""
    u1 = max(random.random(), 1e-10)
    u2 = random.random()
    r = math.sqrt(-2 * math.log(u1))
    theta = 2 * math.pi * u2
    return r * math.cos(theta)


def generate_clusters(
    centers: list[tuple[float, float]],
    points_per_cluster: int,
    spread: float = 1.0,
    seed: int = 42,
) -> list[list[float]]:
    """Генерирует точки вокруг заданных центров."""
    random.seed(seed)
    points: list[list[float]] = []

    for cx, cy in centers:
        for _ in range(points_per_cluster):
            points.append([
                cx + normal_sample() * spread,
                cy + normal_sample() * spread,
            ])

    return points


def print_results(
    points: list[list[float]],
    labels: list[int],
    centroids: list[list[float]],
    iterations: int,
) -> None:
    print("=" * 60)
    print("Демонстрация K-means")
    print("=" * 60)
    print(f"Число точек:     {len(points)}")
    print(f"Число кластеров: {len(centroids)}")
    print(f"Итераций:        {iterations}")
    print(f"Inertia (WCSS):  {inertia(points, labels, centroids):.4f}")
    print()

    cluster_sizes: dict[int, int] = {}
    for label in labels:
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    print("Размеры кластеров:")
    for cluster_id in sorted(cluster_sizes):
        print(f"  Кластер {cluster_id}: {cluster_sizes[cluster_id]} точек")

    print()
    print("Центроиды:")
    for i, centroid in enumerate(centroids):
        print(f"  C{i}: ({centroid[0]:7.3f}, {centroid[1]:7.3f})")

    print()
    print("Первые 10 точек и их метки:")
    for i in range(min(10, len(points))):
        x, y = points[i]
        print(f"  Точка {i:2d}: ({x:7.3f}, {y:7.3f}) -> кластер {labels[i]}")

    print("=" * 60)


def try_visualization(
    points: list[list[float]],
    labels: list[int],
    centroids: list[list[float]],
) -> None:
    """Строит график, если установлен matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен — график пропущен.")
        print("Установите: pip install matplotlib")
        return

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    plt.figure(figsize=(8, 6))

    for cluster_id in range(len(centroids)):
        cluster_points = [p for p, label in zip(points, labels) if label == cluster_id]
        xs = [p[0] for p in cluster_points]
        ys = [p[1] for p in cluster_points]
        plt.scatter(xs, ys, c=colors[cluster_id % len(colors)], alpha=0.6, label=f"Кластер {cluster_id}")

    cx = [c[0] for c in centroids]
    cy = [c[1] for c in centroids]
    plt.scatter(cx, cy, c="black", marker="X", s=200, linewidths=2, label="Центроиды")

    plt.title("K-means: результат кластеризации")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kmeans_result.png", dpi=120)
    print("\nГрафик сохранён в kmeans_result.png")
    plt.show()


def main() -> None:
    # Три явно разделённых облака точек
    true_centers = [(0.0, 0.0), (8.0, 0.0), (4.0, 7.0)]
    points = generate_clusters(true_centers, points_per_cluster=50, spread=0.8, seed=42)

    k = 3
    labels, centroids, iterations = k_means(points, k=k, max_iterations=100, seed=42)

    print_results(points, labels, centroids, iterations)

    # Проверка: точки ближайшего кластера совпадают с assign_clusters
    recomputed = assign_clusters(points, centroids)
    assert recomputed == labels, "Метки кластеров не согласованы"

    try_visualization(points, labels, centroids)
    del_temp_files()


if __name__ == "__main__":
    main()
