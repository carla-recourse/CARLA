import tensorflow as tf


def distance_func(name, x1, x2, eps: float = 0.0):
    if name == "l1":
        ax = 1
        return l1_dist(x1, x2, ax, eps)
    if name == "l2":
        ax = 1
        return l2_dist(x1, x2, ax, eps)
    if name == "cosine":
        ax = -1
        return cosine_dist(x1, x2, ax, eps)


def l1_dist(x1, x2, ax: int, eps: float = 0.0):
    # sum over |x| + eps, i.e. L1 norm
    x = x1 - x2
    return tf.reduce_sum(tf.abs(x), axis=ax) + eps


def l2_dist(x1, x2, ax: int, eps: float = 0.0):
    # sqrt((sum over x^2) + eps)), i.e. L2 norm
    x = x1 - x2
    return (tf.reduce_sum(x**2, axis=ax) + eps) ** 0.5


def cosine_dist(x1, x2, ax: int, eps: float = 0.0):
    # normalize by sqrt(max(sum(x**2), 1e-12))
    normalize_x1 = tf.nn.l2_normalize(x1, dim=1)
    normalize_x2 = tf.nn.l2_normalize(x2, dim=1)
    dist = (
        tf.losses.cosine_distance(
            normalize_x1,
            normalize_x2,
            axis=ax,
            reduction=tf.compat.v1.losses.Reduction.NONE,
        )
        + eps
    )
    dist = tf.squeeze(dist)
    dist = tf.cast(dist, tf.float64)
    return dist
