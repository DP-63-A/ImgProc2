"""
- file reading
- making it gray (avg/HSV-V)
- binarisation(threshold/Otsu)
- normalising, equalising, CLAHE, contrast stretch
- filters (blur, unsharp, laplacian, sobel)
- geometry (cyclic shift, rotate)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from collections import deque
import math


def load_color_image(path: str) -> np.ndarray:
    """Load an image from path as a BGR color numpy array.
    Raises FileNotFoundError if reading fails. Returns an array with shape
    (H, W, 3) and dtype uint8 when successful.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read: {path}")
    return img


def to_gray_average(bgr_img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale by averaging the three color channels.
    This is a simple ungeighted average (not the luminance-weighted
    conversion). Returns a uint8 2D array with values in 0..255.
    """
    avg = np.mean(bgr_img, axis=2)
    return np.clip(avg, 0, 255).astype(np.uint8)


def to_gray_hsv_value(bgr_img: np.ndarray) -> np.ndarray:
    """Convert BGR to HSV and return the V (value/brightness) channel.
    This produces a grayscale image representing brightness rather than
    averaged color intensity.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2]


def binarize_threshold(gray_img: np.ndarray, thresh: int = 127, invert: bool = False) -> np.ndarray:
    """Apply a fixed threshold to a grayscale image.
    If invert is true - the binary output is inverted. Input will be
    converted to uint8 if necessary. Returns a binary image with values
    0 or 255.
    """
    if gray_img.dtype != np.uint8:
        gray = np.clip(gray_img, 0, 255).astype(np.uint8)
    else:
        gray = gray_img
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, thresh, 255, flag)
    return bw


def binarize_otsu(gray_img: np.ndarray, apply_blur: bool = True, blur_ksize: int = 5) -> Tuple[int, np.ndarray]:
    """Compute Otsu's threshold and return the threshold value and binary map.
    If apply_blur is True a Gaussian blur with kernel size blur_ksize
    (adjusted to be odd) is applied before thresholding to reduce noise.
    Returns (threshold_value, binary_image).
    """
    if gray_img.dtype != np.uint8:
        gray = np.clip(gray_img, 0, 255).astype(np.uint8)
    else:
        gray = gray_img
    work = gray
    if apply_blur:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        work = cv2.GaussianBlur(gray, (k, k), 0)
    thresh_val, bw = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thresh_val), bw


def normalize_image(img: np.ndarray, out_min: int = 0, out_max: int = 255) -> np.ndarray:
    """Linearly normalize image intensities to the range [out_min, out_max].
    Works with single- or multi-channel images. Returns uint8 data.
    """
    arr = img.astype(np.float32)
    normed = cv2.normalize(arr, None, alpha=out_min, beta=out_max, norm_type=cv2.NORM_MINMAX)
    return np.clip(normed, out_min, out_max).astype(np.uint8)


def equalize_histogram(gray_img: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization to a grayscale image.
    Input is converted to uint8 if necessary and the equalized uint8 result
    is returned.
    """
    if gray_img.dtype != np.uint8:
        gray = np.clip(gray_img, 0, 255).astype(np.uint8)
    else:
        gray = gray_img
    return cv2.equalizeHist(gray)


def clahe_equalize(gray_img: np.ndarray, clipLimit: float = 2.0, tileGridSize: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE (adaptive histogram equalization) to enhance contrast.
    clipLimit and tileGridSize control the CLAHE parameters. Returns a
    uint8 image.
    """
    if gray_img.dtype != np.uint8:
        gray = np.clip(gray_img, 0, 255).astype(np.uint8)
    else:
        gray = gray_img
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)


def contrast_stretch(gray_img: np.ndarray, low_percent: float = 2.0, high_percent: float = 98.0) -> np.ndarray:
    """Contrast stretch using percentile clamps.
    Values below the low_percent percentile map to 0, above the
    high_percent percentile map to 255, with linear scaling in between.
    Returns uint8 result. If low == high returns a copy of the input.
    """
    if gray_img.dtype != np.uint8:
        gray = np.clip(gray_img, 0, 255).astype(np.uint8)
    else:
        gray = gray_img

    low = np.percentile(gray, low_percent)
    high = np.percentile(gray, high_percent)
    if high == low:
        return gray.copy()
    stretched = (gray.astype(np.float32) - low) * (255.0 / (high - low))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Apply Gaussian blur to the image.
    Ensures the kernel size is odd. Works for single- and multi-channel
    images and returns the blurred image.
    """
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def unsharp_mask(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """Sharpen image using the unsharp mask technique.
    Creates a blurred version of the image, computes a mask (original -
    blurred) and adds a scaled mask back to the original. If threshold > 0
    low-contrast areas are left unchanged to avoid amplifying noise.
    """
    blurred = gaussian_blur(img, kernel_size, sigma)
    img_f = img.astype(np.float32)
    blurred_f = blurred.astype(np.float32)
    mask = img_f - blurred_f
    sharpened = img_f + amount * mask
    if threshold > 0:
        low_contrast_mask = np.absolute(mask) < threshold
        sharpened[low_contrast_mask] = img_f[low_contrast_mask]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


def laplacian_sharpen(img: np.ndarray, ksize: int = 3, alpha: float = 1.0) -> np.ndarray:
    """Enhance sharpness by subtracting a scaled Laplacian (edge boost).
    For color images the operation is applied per-channel and channels are
    recombined. alpha controls sharpening strength.
    """
    if img.ndim == 3:
        channels = [laplacian_sharpen(img[:, :, c], ksize=ksize, alpha=alpha) for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    """
    single-channel
    """
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=ksize)
    """
    subtract Laplacian to boost sharpness
    """
    result = img.astype(np.float32) - alpha * lap
    return np.clip(result, 0, 255).astype(np.uint8)


def sobel_edges(gray_img: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel gradients and magnitude for a grayscale image.
    Returns a tuple (sobel_x_u8, sobel_y_u8, magnitude_u8), where each
    component is normalized to uint8 (0..255) for convenient display. The
    function accepts non-uint8 inputs and converts them as needed.
    """
    if gray_img.dtype != np.uint8:
        gray = np.clip(gray_img, 0, 255).astype(np.uint8)
    else:
        gray = gray_img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    gx_u = np.clip((gx - gx.min()) / (gx.max() - gx.min() + 1e-9) * 255.0, 0, 255).astype(np.uint8)
    gy_u = np.clip((gy - gy.min()) / (gy.max() - gy.min() + 1e-9) * 255.0, 0, 255).astype(np.uint8)
    mag_u = np.clip(mag / (mag.max() + 1e-9) * 255.0, 0, 255).astype(np.uint8)
    return gx_u, gy_u, mag_u


def cyclic_shift(img: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    """Perform a cyclic (wrap-around) shift of the image pixels.
    Positive shift_x moves content to the right; positive shift_y moves
    content down. Wrapping is handled by numpy.roll.
    """
    return np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)


def rotate_image(img: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None, scale: float = 1.0,
                 borderMode=cv2.BORDER_CONSTANT) -> np.ndarray:
    """Rotate an image by (angle) degrees about center with optional scaling.
    The returned image has the same width/height as the input. borderMode
    controls how pixels outside the source area are handled.
    """
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=borderMode)


def overlay_edges(bgr_img: np.ndarray, edges_gray: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), alpha: float = 0.6,
                  edge_threshold: int = 20) -> np.ndarray:
    """Overlays an edge map (grayscale) onto a BGR image using a color mask.
    Pixels in edges_gray above edge_threshold are colored with color
    and blended over bgr_img with opacity alpha. Expects a 3-channel
    BGR background image and returns a uint8 color image.
    """
    if bgr_img.ndim != 3:
        raise ValueError("overlay_edges expects a color image (BGR).")
    mask = edges_gray > edge_threshold
    color_mask = np.zeros_like(bgr_img)
    color_mask[mask] = color
    out = cv2.addWeighted(bgr_img.astype(np.float32), 1.0, color_mask.astype(np.float32), alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def detect_hough_lines(bgr_img: np.ndarray,
                       edge_thresh1: int = 50,
                       edge_thresh2: int = 150,
                       min_line_length: int = 50,
                       max_line_gap: int = 10,
                       draw_color: Tuple[int,int,int] = (0,0,255),
                       thickness: int = 2) -> np.ndarray:
    """
    Detect lines with probabilistic Hough on a grayscale edge map.
    Returns a BGR image with lines drawn.
    """
    if bgr_img is None:
        raise ValueError("detect_hough_lines: input is None")
    src = bgr_img.copy()
    gray = src if src.ndim == 2 else to_gray_average(src)
    edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    out = src.copy() if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)), draw_color, thickness, lineType=cv2.LINE_AA)
    return out


def detect_hough_circles(bgr_img: np.ndarray,
                         dp: float = 1.2,
                         min_dist: float = 20,
                         param1: int = 100,
                         param2: int = 30,
                         min_radius: int = 0,
                         max_radius: int = 0,
                         draw_color: Tuple[int,int,int] = (0,255,0),
                         thickness: int = 2) -> np.ndarray:
    """
    Detect circles using HoughCircles. Works on grayscale (blur recommended).
    Returns color image with circles drawn.
    """
    if bgr_img is None:
        raise ValueError("detect_hough_circles: input is None")
    src = bgr_img.copy()
    gray = src if src.ndim == 2 else to_gray_hsv_value(src)
    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    out = src.copy() if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(out, (x, y), r, draw_color, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 2, (0,0,255), 3)
    return out


def local_stat_features(gray_img: np.ndarray, win: int = 15, compute_entropy: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute local mean, variance, std using box filters.
    Optionally compute local entropy (slow naive histogram method).
    Inputs:
        gray_img - uint8 2D
        win - odd window size
    Returns dict with 'mean', 'var', 'std', and optional 'entropy' maps (float32).
    """
    if gray_img.dtype != np.uint8:
        gray = np.clip(gray_img,0,255).astype(np.uint8)
    else:
        gray = gray_img
    k = win if win % 2 == 1 else win + 1
    mean = cv2.boxFilter(gray.astype(np.float32), ddepth=-1, ksize=(k,k), normalize=True)
    sq = (gray.astype(np.float32) ** 2)
    mean_sq = cv2.boxFilter(sq, ddepth=-1, ksize=(k,k), normalize=True)
    var = mean_sq - (mean ** 2)
    var = np.clip(var, 0, None)
    std = np.sqrt(var)
    out = {
        "mean": mean.astype(np.float32),
        "var": var.astype(np.float32),
        "std": std.astype(np.float32)
    }
    if compute_entropy:
        H, W = gray.shape
        ent = np.zeros((H,W), dtype=np.float32)
        pad = k//2
        padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
        bins = 32
        for y in range(H):
            row = padded[y:y+k, :]
            for x in range(W):
                patch = row[:, x:x+k].ravel()
                hist, _ = np.histogram(patch, bins=bins, range=(0,256), density=True)
                h = hist[hist>0]
                ent[y,x] = -np.sum(h * np.log2(h + 1e-12))
        out["entropy"] = ent
    return out


def region_grow_texture_seed(gray_img: np.ndarray,
                             seed_pt: Tuple[int,int],
                             win: int = 15,
                             dist_thresh: float = 1.5,
                             connectivity: int = 4,
                             max_pixels: int = 1_000_000) -> np.ndarray:
    """
    Region growing segmentation from seed based on local mean/std features.
    - gray_img: uint8 gray
    - seed_pt: (x,y) coordinate in image coords (col,row)
    - win: window size for local features
    - dist_thresh: threshold in feature-space (euclidean after normalization)
    Returns binary mask (uint8 0/255).
    """
    if gray_img.dtype != np.uint8:
        gray = np.clip(gray_img,0,255).astype(np.uint8)
    else:
        gray = gray_img
    H,W = gray.shape
    feats = local_stat_features(gray, win=win, compute_entropy=False)
    mean = feats["mean"]
    std = feats["std"]
    mean_f = (mean - mean.mean()) / (mean.std() + 1e-9)
    std_f = (std - std.mean()) / (std.std() + 1e-9)
    sx, sy = seed_pt
    if not (0 <= sx < W and 0 <= sy < H):
        raise ValueError("seed outside image")
    seed_vec = np.array([mean_f[sy, sx], std_f[sy, sx]], dtype=np.float32)
    visited = np.zeros((H,W), dtype=np.uint8)
    mask = np.zeros((H,W), dtype=np.uint8)
    dq = deque()
    dq.append((sx, sy))
    visited[sy, sx] = 1
    if connectivity == 8:
        neigh = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    else:
        neigh = [(0,-1),(-1,0),(1,0),(0,1)]
    count = 0
    while dq:
        x,y = dq.popleft()
        vec = np.array([mean_f[y,x], std_f[y,x]], dtype=np.float32)
        dist = np.linalg.norm(vec - seed_vec)
        if dist <= dist_thresh:
            mask[y,x] = 255
            count += 1
            if count >= max_pixels:
                break
            for dx,dy in neigh:
                nx, ny = x+dx, y+dy
                if 0 <= nx < W and 0 <= ny < H and not visited[ny,nx]:
                    visited[ny,nx] = 1
                    dq.append((nx, ny))
    return mask