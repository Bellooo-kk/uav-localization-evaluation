import math

import cv2
import numpy as np


def estimate_query_gsd(image_shape, relative_height, pitch_deg, sensor_diagonal, focal_length):
    height, width = image_shape[:2]
    sine = math.sin(math.radians(-pitch_deg))
    if not all(math.isfinite(v) and v > 0 for v in
               (height, width, relative_height, sensor_diagonal, focal_length, sine)):
        raise ValueError("GSD requires a downward-looking image and positive camera inputs.")
    return relative_height * sensor_diagonal / (sine * focal_length * math.hypot(width, height))


def adaptive_tile_size(query_shape, query_gsd, reference_gsd):
    if not all(math.isfinite(v) and v > 0 for v in (query_gsd, reference_gsd)):
        raise ValueError("Both ground sampling distances must be positive and finite.")
    side = min(query_shape[:2])
    if side <= 0:
        raise ValueError("The query image must be nonempty.")
    pixels = math.ceil(side * query_gsd / reference_gsd)
    return pixels + pixels % 2


def tile_windows(reference_image, query_shape, query_gsd, reference_gsd, overlap_percent):
    height, width = reference_image.shape[:2]
    size = adaptive_tile_size(query_shape, query_gsd, reference_gsd)
    if not math.isfinite(overlap_percent) or not 0 <= overlap_percent < 100:
        raise ValueError("Overlap must be a percentage in [0, 100).")
    step = max(1, int(size * (100 - overlap_percent) / 100))
    if size > min(height, width):
        return

    channel = reference_image if reference_image.ndim == 2 else reference_image[:, :, 0]
    mask = cv2.resize((channel > 0).astype(np.uint8),
                      (max(1, width // 10), max(1, height // 10)),
                      interpolation=cv2.INTER_NEAREST)

    def starts(length):
        last = max(0, length - size - 1)
        yield from range(0, last, step)
        yield last

    for top in starts(height):
        for left in starts(width):
            right, bottom = left + size, top + size
            area = mask[top // 10:bottom // 10, left // 10:right // 10]
            if not area.size:
                area = channel[top:bottom, left:right] > 0
                valid = np.count_nonzero(area) >= area.size * 0.4
            else:
                valid = np.count_nonzero(area) >= size * size / 100 * 0.4
            if valid:
                yield left, top, right, bottom


def align_reference(image, yaw_deg):
    height, width = image.shape[:2]
    radians = math.radians(yaw_deg)
    sine, cosine = abs(math.sin(radians)), abs(math.cos(radians))
    aligned_height = int(width * sine + height * cosine)
    aligned_width = int(height * sine + width * cosine)
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), yaw_deg, 1)
    matrix[:, 2] += [(aligned_width - width) // 2, (aligned_height - height) // 2]
    aligned = cv2.warpAffine(image, matrix, (aligned_width, aligned_height))
    return aligned, cv2.invertAffineTransform(matrix)
