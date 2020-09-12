import tensorflow as tf


def cubic(w, a0, a1, a2, a3):
    return a1 + w * (0.5 * a2 - 0.5 * a0) + w * w * (a0 - 2.5 * a1 + 2 * a2 - 0.5 * a3) + w * w * w * (1.5 * a1 - 1.5 * a2 + 0.5 * a3 - 0.5 * a0)
