from __future__ import annotations

import numpy as np

# 2D Symmetries (D_4)
T2 = [
    # Rotations
    np.array([[1, 0], [0, 1]]),  # Identity
    np.array([[0, -1], [1, 0]]),  # 90 degrees rotation
    np.array([[-1, 0], [0, -1]]),  # 180 degrees rotation
    np.array([[0, 1], [-1, 0]]),  # 270 degrees rotation
    # Reflectionsx
    np.array([[1, 0], [0, -1]]),  # Reflect over x-axis
    np.array([[-1, 0], [0, 1]]),  # Reflect over y-axis
    np.array([[0, 1], [1, 0]]),  # Reflect over y=x
    np.array([[0, -1], [-1, 0]]),  # Reflect over y=-x
]

# 3D Symmetries
T3 = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Identity
    # Rotations around x-axis
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # 90 degrees
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # 180 degrees
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # 270 degrees
    # Rotations around y-axis
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # 90 degrees
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 180 degrees
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # 270 degrees
    # Rotations around z-axis
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90 degrees
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 degrees
    np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 270 degrees
    # Reflections
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # Reflect over xy-plane
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # Reflect over yz-plane
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Reflect over xz-plane
]


def pivot(walk: np.ndarray, iterations: int = 1, dimensions: int = 2) -> np.ndarray:
    """
    Applies the pivot algorithm to a self-avoiding walk.

    Parameters:
    walk (np.ndarray): The walk to apply the pivot algorithm to.
    iterations (int, optional): The number of iterations (samples) to perform. Defaults to 1.
    dimensions (int, optional): The number of dimensions for the walk. Defaults to 2.

    Returns:
    np.ndarray: The walk after applying the pivot algorithm.
    """
    T = T2 if dimensions == 2 else T3
    for _ in range(iterations):
        i, m = np.random.randint(len(walk)), T[np.random.randint(len(T))]
        pivot_point = walk[i]
        transformed_segment = (
            m @ (walk[i + 1 :].T - pivot_point[:, None]) + pivot_point[:, None]
        )
        new_walk = np.vstack([walk[: i + 1], transformed_segment.T])
        if len(np.unique(new_walk, axis=0)) == len(new_walk):
            walk = new_walk
    return walk


def rod(n: int, dimensions: int = 2) -> np.ndarray:
    """
    Generates a horizontal rod of length n in 2D or 3D.

    Parameters:
    n (int): The length of the walk to generate.
    dimensions (int, optional): The number of dimensions for the walk. Defaults to 2.

    Returns:
    np.ndarray: The generated walk.
    """
    if dimensions == 2:
        return np.array([[i, 0] for i in range(n)])
    elif dimensions == 3:
        return np.array([[i, 0, 0] for i in range(n)])
    else:
        raise ValueError("Dimensions must be 2 or 3")
