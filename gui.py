import io
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import PySimpleGUI as sg

from image_processing import (
    load_color_image, to_gray_average, to_gray_hsv_value,
    binarize_threshold, binarize_otsu,
    normalize_image, equalize_histogram, clahe_equalize, contrast_stretch,
    gaussian_blur, unsharp_mask, laplacian_sharpen, sobel_edges,
    cyclic_shift, rotate_image, overlay_edges, 
    detect_hough_lines, detect_hough_circles, local_stat_features, region_grow_texture_seed
)


_seed_tmp = {"pt": None}
def _on_mouse_pick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _seed_tmp["pt"] = (x,y)
        cv2.destroyWindow("Pick seed")

def pick_seed_point_cv(img: np.ndarray, window_name: str = "Pick seed") -> Optional[Tuple[int,int]]:
    """
    Show a window, wait for single left click, return (x,y) or none.
    This blocks until click or window closed.
    """
    if img is None:
        return None
    disp = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, disp)
    _seed_tmp["pt"] = None
    cv2.setMouseCallback(window_name, _on_mouse_pick)
    while True:
        k = cv2.waitKey(50) & 0xFF
        if _seed_tmp["pt"] is not None:
            pt = _seed_tmp["pt"]
            cv2.destroyWindow(window_name)
            return pt
        if k == 27:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass
            return None



def cv2_to_bytes(img: np.ndarray, maxsize: Tuple[int, int] = (900, 600)) -> bytes:
    """Convert a cv2 image (BGR or grayscale) to PNG bytes for PySimpleGUI.Image.
    The function converts BGR images to RGB (and grayscale to RGB), creates a
    Pillow image, resizes it to fit inside maxsize while preserving aspect
    ratio, and returns PNG-encoded bytes suitable for the `Image` element.
    Returns empty bytes for a None input.
    """
    if img is None:
        return b""
    if img.ndim == 2:
        arr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(arr)
    pil.thumbnail(maxsize, Image.LANCZOS)
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    return bio.getvalue()


def safe_int(val, default=0):
    """Safely convert val to int, returning default on failure.
    Uses a broad exception catch to avoid GUI crashes when parsing user input.
    """
    try:
        return int(val)
    except Exception:
        return default


def safe_float(val, default=0.0):
    """Safely convert val to float, returning default on failure.
    Used when reading numeric parameters from the GUI controls.
    """
    try:
        return float(val)
    except Exception:
        return default


FUNC_LIST = [
    "None",
    "Gray(avg)",
    "Gray(HSV V)",
    "Binarize Threshold",
    "Binarize Otsu",
    "Normalize",
    "Equalize",
    "CLAHE",
    "Contrast Stretch",
    "Gaussian Blur",
    "Unsharp Mask",
    "Laplacian Sharpen",
    "Sobel Edges",
    "Cyclic Shift",
    "Rotate",
    "Overlay Edges",
    "Hough Lines",
    "Hough Circles",
    "Local Stats",
    "Seed Texture Segment",
    "Reset"
]

sg.theme("SystemDefault")

"""
Left column for parameters
"""
left = [
    [sg.Button("Load", key="-LOAD-"), sg.Button("Save As", key="-SAVE-"), sg.Button("Reset", key="-RESET-"), sg.Button("Undo", key="-UNDO-"), sg.Button("Redo", key="-REDO-")],
    [sg.Text("Function:"), sg.Combo(FUNC_LIST, default_value="None", key="-FUNC-", size=(30,1), enable_events=True), sg.Button("Apply", key="-APPLY-")],

    [sg.HorizontalSeparator()],
    [sg.Text("Parameters:")],

    [sg.Text("Threshold:"), sg.Slider(range=(0,255), orientation="h", key="-P-THR-", default_value=127, visible=False)],
    [sg.Checkbox("Invert", key="-P-INVERT-", visible=False)],
    
    [sg.Checkbox("Blur before Otsu", default=True, key="-P-OTSU-BLUR-", visible=False)],
    [sg.Text("Otsu blur ksize:"), sg.Spin([3,5,7,9], initial_value=5, key="-P-OTSU-K-", visible=False)],
    
    [sg.Text("CLAHE clip:"), sg.Slider(range=(1,10), resolution=0.5, orientation="h", key="-P-CLAHE-CLIP-", default_value=2.0, visible=False)],
    [sg.Text("CLAHE tiles (w,h):"), sg.Input("8,8", key="-P-CLAHE-TILE-", size=(12,1), visible=False)],
    
    [sg.Text("Stretch low %:"), sg.Slider(range=(0,10), orientation="h", key="-P-STRETCH-LOW-", default_value=2, visible=False)],
    [sg.Text("Stretch high %:"), sg.Slider(range=(90,100), orientation="h", key="-P-STRETCH-HIGH-", default_value=98, visible=False)],
    
    [sg.Text("Blur ksize:"), sg.Spin([3,5,7,9,11], initial_value=5, key="-P-BLUR-K-", visible=False)],
    [sg.Text("Blur sigma:"), sg.Slider(range=(0.0,10.0), resolution=0.1, orientation="h", key="-P-BLUR-SIG-", default_value=0.0, visible=False)],
    
    [sg.Text("Unsharp k:"), sg.Spin([3,5,7,9], initial_value=5, key="-P-US-K-", visible=False)],
    [sg.Text("Unsharp sigma:"), sg.Slider(range=(0.1,5.0), resolution=0.1, orientation="h", key="-P-US-SIG-", default_value=1.0, visible=False)],
    [sg.Text("Unsharp amount:"), sg.Slider(range=(0.0,3.0), resolution=0.1, orientation="h", key="-P-US-AMT-", default_value=1.0, visible=False)],
    [sg.Text("Unsharp threshold:"), sg.Slider(range=(0,50), orientation="h", key="-P-US-THR-", default_value=0, visible=False)],
    
    [sg.Text("Lap k:"), sg.Spin([1,3,5], initial_value=3, key="-P-LAP-K-", visible=False)],
    [sg.Text("Lap alpha:"), sg.Slider(range=(0.1,3.0), resolution=0.1, orientation="h", key="-P-LAP-A-", default_value=1.0, visible=False)],
    
    [sg.Text("Sobel k:"), sg.Spin([1,3,5], initial_value=3, key="-P-SOBEL-K-", visible=False)],
    
    [sg.Text("Shift X:"), sg.Input("0", key="-P-SHIFT-X-", size=(6,1), visible=False), sg.Text("Shift Y:"), sg.Input("0", key="-P-SHIFT-Y-", size=(6,1), visible=False)],
    
    [sg.Text("Angle:"), sg.Slider(range=(-180,180), orientation="h", key="-P-ROT-A-", default_value=15, visible=False)],
    
    [sg.Text("Edge thr:"), sg.Slider(range=(0,100), orientation="h", key="-P-EDGE-TH-", default_value=20, visible=False)],
    [sg.Text("Overlay alpha:"), sg.Slider(range=(0.0,1.0), resolution=0.05, orientation="h", key="-P-EDGE-ALPHA-", default_value=0.6, visible=False)],
    [sg.HorizontalSeparator()],
    [sg.Text("", key="-STATUS-", size=(40,2))],

    [sg.Text("Hough minLineLen:"), sg.Input("50", key="-P-HL-MINL-", size=(6,1), visible=False),
     sg.Text("maxGap:"), sg.Input("10", key="-P-HL-MAXG-", size=(6,1), visible=False)],
    [sg.Text("Hough Canny T1:"), sg.Input("50", key="-P-HL-T1-", size=(6,1), visible=False),
     sg.Text("T2:"), sg.Input("150", key="-P-HL-T2-", size=(6,1), visible=False)],

    [sg.Text("Hough dp:"), sg.Input("1.2", key="-P-HC-DP-", size=(6,1), visible=False),
     sg.Text("minDist:"), sg.Input("20", key="-P-HC-MIND-", size=(6,1), visible=False)],
    [sg.Text("param1:"), sg.Input("100", key="-P-HC-P1-", size=(6,1), visible=False),
     sg.Text("param2:"), sg.Input("30", key="-P-HC-P2-", size=(6,1), visible=False)],
    [sg.Text("minR:"), sg.Input("0", key="-P-HC-MINR-", size=(6,1), visible=False),
     sg.Text("maxR:"), sg.Input("0", key="-P-HC-MAXR-", size=(6,1), visible=False)],

    [sg.Text("Local win:"), sg.Spin([3,5,7,9,11,15,21], initial_value=15, key="-P-LW-", visible=False)],
    [sg.Text("Seg dist thr:"), sg.Slider(range=(0.1,30.0), resolution=0.1, orientation="h", key="-P-SEG-TH-", default_value=6.0, visible=False)],
    [sg.Button("Pick Seed (click image)", key="-PICK-SEED-", visible=False), sg.Text("Seed:(-, -)", key="-SEED-COORD-", visible=False)],

]

image_col = [
    [sg.Image(data=None, key="-IMAGE-", size=(900,600))],
    [sg.Text("History (latest last):"), sg.Listbox(values=[], size=(80,6), key="-HISTORY-")]
]

layout = [
    [sg.Column(left), sg.VerticalSeparator(), sg.Column(image_col)]
]

window = sg.Window("Stage2 Image Tool (fixed)", layout, resizable=True, finalize=True)

"""
State
"""
original_image = None
current_image = None
history_images = []   
history_labels = []
history_index = -1    

MAX_HISTORY = 20

def push_state(img: np.ndarray, label: str):
    """Store a new image state in the undo/redo history and update the UI.
    This function appends a copy of img together with label to the
    history_images/history_labels lists. If the user previously undid
    actions (i.e. history_index is not at the end), any redo states beyond
    the current index are discarded. The history size is capped by
    MAX_HISTORY; older entries are removed when the cap is exceeded. The
    GUI listbox is updated to reflect the new history labels.
    """
    global history_images, history_labels, history_index
    """
    if we are not at the end, drop redo history
    """
    if history_index < len(history_images) - 1:
        history_images = history_images[:history_index+1]
        history_labels = history_labels[:history_index+1]
    history_images.append(img.copy())
    history_labels.append(label)
    """
    keep limit
    """
    if len(history_images) > MAX_HISTORY:
        history_images.pop(0)
        history_labels.pop(0)
    history_index = len(history_images) - 1
    window["-HISTORY-"].update(history_labels)

def set_current_from_history():
    """Load the image at history_index into current_image and refresh UI.
    Copies the image from the history list into current_image, updates the
    main -IMAGE- widget (converting to PNG bytes) and writes a brief status
    message with the image size and history index.
    """
    global current_image, history_index
    if 0 <= history_index < len(history_images):
        current_image = history_images[history_index].copy()
        window["-IMAGE-"].update(data=cv2_to_bytes(current_image))
        window["-STATUS-"].update(f"Image size: {current_image.shape[1]}x{current_image.shape[0]} (history idx {history_index})")

def show_params_for(fn_name: str):
    """Hide all parameter controls then enable only those relevant to fn_name.
    The GUI pre-creates a long list of parameter widgets. This helper hides
    them all first and then selectively makes visible the controls required
    for the chosen operation (e.g. blur parameters for Gaussian Blur).
    """
    all_keys = [
        "-P-THR-", "-P-INVERT-", "-P-OTSU-BLUR-", "-P-OTSU-K-",
        "-P-CLAHE-CLIP-", "-P-CLAHE-TILE-",
        "-P-STRETCH-LOW-", "-P-STRETCH-HIGH-",
        "-P-BLUR-K-", "-P-BLUR-SIG-",
        "-P-US-K-", "-P-US-SIG-", "-P-US-AMT-", "-P-US-THR-",
        "-P-LAP-K-", "-P-LAP-A-",
        "-P-SOBEL-K-",
        "-P-SHIFT-X-", "-P-SHIFT-Y-",
        "-P-ROT-A-",
        "-P-EDGE-TH-", "-P-EDGE-ALPHA-"
    ]
    for k in all_keys:
        try:
            window[k].update(visible=False)
        except Exception:
            pass

    if fn_name == "Binarize Threshold":
        for k in ("-P-THR-", "-P-INVERT-"):
            window[k].update(visible=True)
    elif fn_name == "Binarize Otsu":
        for k in ("-P-OTSU-BLUR-", "-P-OTSU-K-"):
            window[k].update(visible=True)
    elif fn_name == "CLAHE":
        for k in ("-P-CLAHE-CLIP-", "-P-CLAHE-TILE-"):
            window[k].update(visible=True)
    elif fn_name == "Contrast Stretch":
        for k in ("-P-STRETCH-LOW-", "-P-STRETCH-HIGH-"):
            window[k].update(visible=True)
    elif fn_name == "Gaussian Blur":
        for k in ("-P-BLUR-K-", "-P-BLUR-SIG-"):
            window[k].update(visible=True)
    elif fn_name == "Unsharp Mask":
        for k in ("-P-US-K-", "-P-US-SIG-", "-P-US-AMT-", "-P-US-THR-"):
            window[k].update(visible=True)
    elif fn_name == "Laplacian Sharpen":
        for k in ("-P-LAP-K-", "-P-LAP-A-"):
            window[k].update(visible=True)
    elif fn_name == "Sobel Edges":
        window["-P-SOBEL-K-"].update(visible=True)
    elif fn_name == "Cyclic Shift":
        for k in ("-P-SHIFT-X-", "-P-SHIFT-Y-"):
            window[k].update(visible=True)
    elif fn_name == "Rotate":
        window["-P-ROT-A-"].update(visible=True)
    elif fn_name == "Overlay Edges":
        for k in ("-P-EDGE-TH-", "-P-EDGE-ALPHA-"):
            window[k].update(visible=True)
    elif fn_name == "Hough Lines":
        for k in ("-P-HL-MINL-","-P-HL-MAXG-","-P-HL-T1-","-P-HL-T2-"):
            window[k].update(visible=True)
    elif fn_name == "Hough Circles":
        for k in ("-P-HC-DP-","-P-HC-MIND-","-P-HC-P1-","-P-HC-P2-","-P-HC-MINR-","-P-HC-MAXR-"):
            window[k].update(visible=True)
    elif fn_name == "Local Stats":
        window["-P-LW-"].update(visible=True)
    elif fn_name == "Seed Texture Segment":
        for k in ("-P-LW-","-P-SEG-TH-","-PICK-SEED-","-SEED-COORD-"):
            window[k].update(visible=True)


def apply_function(fn_name: str, values):
    """Apply a GUI-selected image-processing operation to the current image.
    This function maps the human-friendly function name chosen in the GUI
    (e.g. Gaussian blur, CLAHE) to the corresponding processing
    function in image_processing.py. It extracts and validates parameters
    from th values dict, converts color images to grayscale where
    appropriate, calls the processing routine, and on success pushes the
    result into the undo/redo history and refreshes the displayed image.
    Errors during processing are caught and displayed in a popup with a
    traceback to help debugging.
    """
    global current_image, original_image, history_index
    if current_image is None:
        sg.popup("Load image first.")
        return
    try:
        src = current_image.copy()
        out = None
        label = fn_name

        if fn_name == "None":
            return
        if fn_name == "Reset":
            if original_image is not None:
                push_state(original_image.copy(), "Reset -> original")
                set_current_from_history()
            return
        if fn_name == "Gray(avg)":
            out = to_gray_average(src) if src.ndim == 3 else src.copy()
        elif fn_name == "Gray(HSV V)":
            out = to_gray_hsv_value(src) if src.ndim == 3 else src.copy()
        elif fn_name == "Binarize Threshold":
            thr = safe_int(values.get("-P-THR-", 127), 127)
            inv = bool(values.get("-P-INVERT-", False))
            gray = to_gray_average(src) if src.ndim == 3 else src
            out = binarize_threshold(gray, thresh=thr, invert=inv)
        elif fn_name == "Binarize Otsu":
            blur_flag = bool(values.get("-P-OTSU-BLUR-", True))
            k = safe_int(values.get("-P-OTSU-K-", 5), 5)
            gray = to_gray_average(src) if src.ndim == 3 else src
            otsu_val, bw = binarize_otsu(gray, apply_blur=blur_flag, blur_ksize=k)
            out = bw
            label += f" (Otsu={otsu_val})"
        elif fn_name == "Normalize":
            gray = to_gray_average(src) if src.ndim == 3 else src
            out = normalize_image(gray)
        elif fn_name == "Equalize":
            gray = to_gray_average(src) if src.ndim == 3 else src
            out = equalize_histogram(gray)
        elif fn_name == "CLAHE":
            clip = safe_float(values.get("-P-CLAHE-CLIP-", 2.0), 2.0)
            tile_raw = values.get("-P-CLAHE-TILE-", "8,8")
            try:
                tile = tuple(int(x.strip()) for x in tile_raw.split(",") if x.strip())
                if len(tile) != 2:
                    tile = (8,8)
            except Exception:
                tile = (8,8)
            gray = to_gray_average(src) if src.ndim == 3 else src
            out = clahe_equalize(gray, clipLimit=clip, tileGridSize=tile)
        elif fn_name == "Contrast Stretch":
            low = safe_float(values.get("-P-STRETCH-LOW-", 2.0), 2.0)
            high = safe_float(values.get("-P-STRETCH-HIGH-", 98.0), 98.0)
            gray = to_gray_average(src) if src.ndim == 3 else src
            out = contrast_stretch(gray, low_percent=low, high_percent=high)
        elif fn_name == "Gaussian Blur":
            k = safe_int(values.get("-P-BLUR-K-", 5), 5)
            s = safe_float(values.get("-P-BLUR-SIG-", 0.0), 0.0)
            out = gaussian_blur(src, ksize=k, sigma=s)
        elif fn_name == "Unsharp Mask":
            k = safe_int(values.get("-P-US-K-", 5), 5)
            s = safe_float(values.get("-P-US-SIG-", 1.0), 1.0)
            amt = safe_float(values.get("-P-US-AMT-", 1.0), 1.0)
            thr = safe_int(values.get("-P-US-THR-", 0), 0)
            out = unsharp_mask(src, kernel_size=k, sigma=s, amount=amt, threshold=thr)
        elif fn_name == "Laplacian Sharpen":
            k = safe_int(values.get("-P-LAP-K-", 3), 3)
            a = safe_float(values.get("-P-LAP-A-", 1.0), 1.0)
            out = laplacian_sharpen(src, ksize=k, alpha=a)
        elif fn_name == "Sobel Edges":
            k = safe_int(values.get("-P-SOBEL-K-", 3), 3)
            gray = to_gray_average(src) if src.ndim == 3 else src
            gx, gy, mag = sobel_edges(gray, ksize=k)
            out = mag
        elif fn_name == "Cyclic Shift":
            sx = safe_int(values.get("-P-SHIFT-X-", 0), 0)
            sy = safe_int(values.get("-P-SHIFT-Y-", 0), 0)
            out = cyclic_shift(src, shift_x=sx, shift_y=sy)
        elif fn_name == "Rotate":
            angle = safe_float(values.get("-P-ROT-A-", 15.0), 15.0)
            out = rotate_image(src, angle=angle)
        elif fn_name == "Overlay Edges":
            th = safe_int(values.get("-P-EDGE-TH-", 20), 20)
            alpha = safe_float(values.get("-P-EDGE-ALPHA-", 0.6), 0.6)
            gray = to_gray_average(src) if src.ndim == 3 else src
            _, _, mag = sobel_edges(gray, ksize=3)
            bg = src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            out = overlay_edges(bg, mag, alpha=alpha, edge_threshold=th)
        elif fn_name == "Hough Lines":
            t1 = safe_int(values.get("-P-HL-T1-", 50), 50)
            t2 = safe_int(values.get("-P-HL-T2-", 150), 150)
            minl = safe_int(values.get("-P-HL-MINL-", 50), 50)
            maxg = safe_int(values.get("-P-HL-MAXG-", 10), 10)
            bg = src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            out = detect_hough_lines(bg, edge_thresh1=t1, edge_thresh2=t2,
                                     min_line_length=minl, max_line_gap=maxg)
        elif fn_name == "Hough Circles":
            dp = float(values.get("-P-HC-DP-", 1.2))
            mind = float(values.get("-P-HC-MIND-", 20))
            p1 = safe_int(values.get("-P-HC-P1-", 100), 100)
            p2 = safe_int(values.get("-P-HC-P2-", 30), 30)
            minr = safe_int(values.get("-P-HC-MINR-", 0), 0)
            maxr = safe_int(values.get("-P-HC-MAXR-", 0), 0)
            bg = src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            out = detect_hough_circles(bg, dp=dp, min_dist=mind, param1=p1, param2=p2, min_radius=minr, max_radius=maxr)
        elif fn_name == "Local Stats":
            win = safe_int(values.get("-P-LW-", 15), 15)
            gray = to_gray_average(src) if src.ndim == 3 else src
            feats = local_stat_features(gray, win=win, compute_entropy=False)
            mean_u = normalize_image(feats["mean"].astype(np.float32))
            std_u = normalize_image(feats["std"].astype(np.float32))
            vis = cv2.merge([mean_u, std_u, np.zeros_like(mean_u)])
            out = vis
        elif fn_name == "Seed Texture Segment":
            win = safe_int(values.get("-P-LW-", 15), 15)
            thr = float(values.get("-P-SEG-TH-", 6.0))
            bg_for_pick = src.copy() if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            pick = pick_seed_point_cv(bg_for_pick, window_name="Pick seed (click)")
            if pick is None:
                sg.popup("Seed pick cancelled.")
                return
            sx, sy = pick
            window["-SEED-COORD-"].update(f"Seed:({sx},{sy})")
            gray = to_gray_average(src) if src.ndim == 3 else src
            mask = region_grow_texture_seed(gray, (sx,sy), win=win, dist_thresh=thr, connectivity=8)
            bg = src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            colored = bg.copy()
            colored[mask==255] = (0,255,0)
            out = cv2.addWeighted(bg.astype(np.float32), 0.6, colored.astype(np.float32), 0.4, 0).astype(np.uint8)
        else:
            sg.popup(f"Function {fn_name} not implemented.")
            return

        if out is not None:
            push_state(out.copy(), label)
            set_current_from_history()

    except Exception as e:
        tb = traceback.format_exc()
        sg.popup_error("Error with using function:", str(e), "\n\nTraceback:\n" + tb)


"""
Event loop
"""
window["-STATUS-"].update("Ready. Load an image to start.")

while True:
    event, values = window.read(timeout=100)
    if event == sg.WIN_CLOSED:
        break

    if event == "-LOAD-":
        filename = sg.popup_get_file("Select image", file_types=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),))
        if filename:
            try:
                img = load_color_image(filename)
                original_image = img.copy()
                history_images = []
                history_labels = []
                history_index = -1
                push_state(original_image.copy(), "Loaded")
                set_current_from_history()
                window["-STATUS-"].update(f"Loaded: {filename}")
            except Exception as e:
                sg.popup_error("Cannot load image:", str(e))

    if event == "-SAVE-":
        if current_image is None:
            sg.popup("No image to save.")
        else:
            save_to = sg.popup_get_file("Save as", save_as=True, file_types=(("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"),), default_extension="png")
            if save_to:
                try:
                    cv2.imwrite(save_to, current_image)
                    sg.popup("Saved:", save_to)
                except Exception as e:
                    sg.popup_error("Save error:", str(e))

    if event == "-RESET-":
        if original_image is not None:
            push_state(original_image.copy(), "Reset -> original")
            set_current_from_history()
        else:
            sg.popup("No original image loaded.")

    if event == "-UNDO-":
        if history_index > 0:
            history_index -= 1
            set_current_from_history()
        else:
            sg.popup("Nothing to undo.")

    if event == "-REDO-":
        if history_index < len(history_images) - 1:
            history_index += 1
            set_current_from_history()
        else:
            sg.popup("Nothing to redo.")

    if event == "-FUNC-":
        fn = values.get("-FUNC-", "None")
        show_params_for(fn)

    if event == "-APPLY-":
        fn = values.get("-FUNC-", "None")
        apply_function(fn, values)    

window.close()