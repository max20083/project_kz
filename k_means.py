"""
K-means clustering — реализация с нуля без готовых функций кластеризации.
"""

from __future__ import annotations

import os
import random
from typing import List, Sequence, Tuple


Point = List[float]
Centroid = List[float]


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Евклидово расстояние между двумя точками."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def mean_point(points: List[Point]) -> Centroid:
    """Среднее арифметическое по каждой координате."""
    if not points:
        raise ValueError("Нельзя вычислить среднее для пустого набора точек")

    dim = len(points[0])
    centroid: Centroid = [0.0] * dim

    for point in points:
        for i in range(dim):
            centroid[i] += point[i]

    n = len(points)
    return [coord / n for coord in centroid]


def assign_clusters(points: List[Point], centroids: List[Centroid]) -> List[int]:
    """Назначает каждой точке ближайший центроид."""
    labels: List[int] = []

    for point in points:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        labels.append(distances.index(min(distances)))

    return labels


def update_centroids(
    points: List[Point],
    labels: List[int],
    k: int,
) -> List[Centroid]:
    """Пересчитывает центроиды как средние точек каждого кластера."""
    new_centroids: List[Centroid] = []

    for cluster_id in range(k):
        cluster_points = [point for point, label in zip(points, labels) if label == cluster_id]

        if cluster_points:
            new_centroids.append(mean_point(cluster_points))
        else:
            # Если кластер опустел, выбираем случайную точку из данных
            new_centroids.append(random.choice(points)[:])

    return new_centroids


def centroids_changed(
    old: List[Centroid],
    new: List[Centroid],
    tolerance: float = 1e-6,
) -> bool:
    """Проверяет, изменились ли центроиды больше допустимого порога."""
    for old_c, new_c in zip(old, new):
        if euclidean_distance(old_c, new_c) > tolerance:
            return True
    return False


def init_centroids(points: List[Point], k: int, seed: int | None = None) -> List[Centroid]:
    """Случайная инициализация: k различных точек из набора данных."""
    if seed is not None:
        random.seed(seed)

    if k > len(points):
        raise ValueError("k не может быть больше числа точек")

    return [point[:] for point in random.sample(points, k)]


def k_means(
    points: List[Point],
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    seed: int | None = None,
) -> Tuple[List[int], List[Centroid], int]:
    """
    Алгоритм K-means.

    Возвращает:
        labels      — метки кластеров для каждой точки
        centroids   — финальные центроиды
        iterations  — число выполненных итераций
    """
    if k <= 0:
        raise ValueError("k должно быть положительным")

    if not points:
        raise ValueError("Набор точек не может быть пустым")

    centroids = init_centroids(points, k, seed=seed)

    for iteration in range(1, max_iterations + 1):
        labels = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, labels, k)

        if not centroids_changed(centroids, new_centroids, tolerance):
            return labels, new_centroids, iteration

        centroids = new_centroids

    return labels, centroids, max_iterations


def inertia(points: List[Point], labels: List[int], centroids: List[Centroid]) -> float:
    """Сумма квадратов расстояний от точек до центроидов их кластеров."""
    total = 0.0

    for point, label in zip(points, labels):
        total += euclidean_distance(point, centroids[label]) ** 2

    return total

def del_temp_files(file = "kmeans_result.png"):
    os.remove(file)
