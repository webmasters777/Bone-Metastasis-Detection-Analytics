import os
from functools import lru_cache

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Dataset path
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset project")

@lru_cache(maxsize=1)
def load_resnet50_model():
    model_path = os.path.join(os.path.dirname(__file__), "bone_scan_resnet50_final.pth")
    # If a download URL is provided via env var, try to fetch weights when missing
    resnet_url = os.environ.get("RESNET_WEIGHTS_URL")
    if resnet_url and not os.path.exists(model_path):
        try:
            download_file(resnet_url, model_path)
        except Exception as _:
            pass

    if os.path.exists(model_path):
        model = models.resnet50()
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return model
    else:
        print("ResNet50 model not found. Please train it first using train_resnet50.py")
        return None

# Load EfficientNet-B3 model
@lru_cache(maxsize=1)
def load_efficientnet_b3_model():
    model_path = os.path.join(os.path.dirname(__file__), "bone_scan_efficientnet_b3_final.pth")
    try:
        # If a download URL is provided via env var, try to fetch weights when missing
        eff_url = os.environ.get("EFFICIENTNET_B3_WEIGHTS_URL")
        if eff_url and not os.path.exists(model_path):
            try:
                download_file(eff_url, model_path)
            except Exception:
                pass

        model = models.efficientnet_b3()
        # build classifier to match training
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            return model
        else:
            print("EfficientNet-B3 model not found. Please train it first using train_efficientnet_b3.py")
            return None
    except Exception as e:
        print("Failed to load EfficientNet-B3 model:", e)
        return None


def download_file(url, dest_path, chunk_size=1 << 20):
    """Download a file from `url` to `dest_path` streaming in chunks.

    Raises an exception on non-200 response.
    """
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as out:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                out.write(chunk)
    os.replace(tmp_path, dest_path)


def iter_image_files(folder_path):
    for root, _dirs, files in os.walk(folder_path):
        for name in sorted(files):
            _, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTENSIONS:
                yield os.path.join(root, name)


def load_image_bgr(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img_bgr


def classify_with_efficientnet_b3(img_bgr):
    """Classify image using EfficientNet-B3 model. Returns dict with prediction and confidence."""
    efficientnet_model = load_efficientnet_b3_model()
    if efficientnet_model is None:
        return {"prediction": -1, "confidence": 0.0, "class": "Unknown"}

    # Preprocess
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    preprocess = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(pil).unsqueeze(0)

    with torch.no_grad():
        out = efficientnet_model(tensor)
        prob = float(out.squeeze().item())
        pred = 1 if prob > 0.5 else 0

    return {"prediction": pred, "probability": prob, "confidence": prob, "class": "Metastasis" if pred == 1 else "Normal"}






def extract_metrics(img_bgr):
    """
    OPTIMIZED: Extract only metrics WITHOUT creating visualization.
    This is fast and efficient for batch processing and metrics-only mode.
    
    Returns:
        dict: Contains all computed metrics and processed images
    """
    if img_bgr is None:
        raise ValueError("Input image is empty or unreadable.")

    # Color space conversions (minimal, only for processing)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Filtering
    median = cv2.medianBlur(gray, 5)
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

    # Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gaussian)
    norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Edge detection
    gx = cv2.Sobel(norm, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))
    canny = cv2.Canny(norm, 50, 150)

    # Thresholding
    otsu_val, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, hotspot_mask = cv2.threshold(norm, int(otsu_val * 1.1), 255, cv2.THRESH_BINARY)

    # Texture features - GLCM
    glcm = graycomatrix(norm, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    # Texture features - LBP
    lbp_map = local_binary_pattern(norm, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp_map.ravel(), bins=26, range=(0, 26), density=True)

    # Statistical metrics
    mean_val = float(norm.mean())
    std_val = float(norm.std())
    median_val = float(np.median(norm))

    metrics = {
        "contrast": float(contrast),
        "energy": float(energy),
        "homogeneity": float(homogeneity),
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "otsu": float(otsu_val),
    }

    # Store processed images for visualization (if needed later)
    processed_data = {
        "original": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "gray": gray,
        "median": median,
        "gaussian": gaussian,
        "enhanced": enhanced,
        "norm": norm,
        "sobel": sobel,
        "canny": canny,
        "thresh": thresh,
        "hotspot_display": cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB),
        "hotspot_mask": hotspot_mask,
        "lbp_hist": lbp_hist,
        "metrics": metrics,
    }

    return processed_data


def create_visualization(processed_data, fname="image"):
    """
    OPTIMIZED: Create visualization from pre-computed data.
    This is called ONLY when visualization is needed.
    
    Args:
        processed_data: dict from extract_metrics()
        fname: filename for title
        
    Returns:
        matplotlib.figure.Figure: The analysis figure
    """
    # Extract data
    original = processed_data["original"]
    gray = processed_data["gray"]
    median = processed_data["median"]
    gaussian = processed_data["gaussian"]
    enhanced = processed_data["enhanced"]
    norm = processed_data["norm"]
    sobel = processed_data["sobel"]
    canny = processed_data["canny"]
    thresh = processed_data["thresh"]
    hotspot_display = processed_data["hotspot_display"]
    hotspot_mask = processed_data["hotspot_mask"]
    lbp_hist = processed_data["lbp_hist"]
    metrics = processed_data["metrics"]
    
    contrast = metrics["contrast"]
    energy = metrics["energy"]
    homogeneity = metrics["homogeneity"]
    mean_val = metrics["mean"]
    std_val = metrics["std"]
    median_val = metrics["median"]
    otsu_val = metrics["otsu"]

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(f"Analysis | {fname}", fontsize=14, fontweight="bold", y=0.98)

    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.3)

    for ax, im, title, cmap in zip(
        [fig.add_subplot(gs[0, i]) for i in range(4)],
        [original, gray, median, gaussian],
        ["Original", "Gray", "Median", "Gaussian"],
        [None, "gray", "gray", "gray"],
    ):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for ax, im, title in zip(
        [fig.add_subplot(gs[1, i]) for i in range(4)],
        [enhanced, norm, sobel, canny],
        ["Enhanced Image", "Normalized", "Sobel", "Canny"],
    ):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    ax_thr = fig.add_subplot(gs[2, 0])
    ax_thr.imshow(thresh, cmap="gray")
    ax_thr.set_title("Threshold", fontsize=10)
    ax_thr.axis("off")

    ax_hot = fig.add_subplot(gs[2, 1])
    ax_hot.imshow(hotspot_display)
    ax_hot.set_title("Hotspot", fontsize=10)
    ax_hot.axis("off")

    ax_h1 = fig.add_subplot(gs[2, 2])
    hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256])
    ax_h1.bar(range(256), hist_orig.ravel(), color="blue", width=1.0, alpha=0.8)
    ax_h1.set_title("Original Histogram", fontsize=10)
    ax_h1.set_xlim([0, 255])
    ax_h1.set_xlabel("Pixel Value", fontsize=8)
    ax_h1.set_ylabel("Frequency", fontsize=8)
    ax_h1.tick_params(labelsize=7)

    ax_h2 = fig.add_subplot(gs[2, 3])
    hist_enh = cv2.calcHist([norm], [0], None, [256], [0, 256])
    ax_h2.bar(range(256), hist_enh.ravel(), color="green", width=1.0, alpha=0.8)
    ax_h2.set_title("Enhanced Histogram", fontsize=10)
    ax_h2.set_xlim([0, 255])
    ax_h2.set_xlabel("Pixel Value", fontsize=8)
    ax_h2.set_ylabel("Frequency", fontsize=8)
    ax_h2.tick_params(labelsize=7)

    ax_glcm = fig.add_subplot(gs[3, 0])
    ax_glcm.axis("off")
    ax_glcm.set_title("GLCM Features", fontsize=10, pad=4)
    tbl1 = ax_glcm.table(
        cellText=[
            ["Contrast", f"{contrast:.4f}"],
            ["Energy", f"{energy:.4f}"],
            ["Homogeneity", f"{homogeneity:.4f}"],
        ],
        colLabels=["Feature", "Value"],
        loc="center",
        cellLoc="center",
    )
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(9)
    tbl1.scale(1.2, 1.6)
    for (row, _col), cell in tbl1.get_celld().items():
        if row == 0:
            cell.set_facecolor("#378ADD")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#E6F1FB" if row % 2 == 0 else "white")

    ax_stat = fig.add_subplot(gs[3, 1])
    ax_stat.axis("off")
    ax_stat.set_title("Statistics", fontsize=10, pad=4)
    tbl2 = ax_stat.table(
        cellText=[
            ["Mean", f"{mean_val:.2f}"],
            ["Std", f"{std_val:.2f}"],
            ["Median", f"{median_val:.2f}"],
            ["Otsu thr", f"{otsu_val:.0f}"],
        ],
        colLabels=["Statistic", "Value"],
        loc="center",
        cellLoc="center",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)
    tbl2.scale(1.2, 1.6)
    for (row, _col), cell in tbl2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1D9E75")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EAF3DE" if row % 2 == 0 else "white")

    ax_lbp = fig.add_subplot(gs[3, 2:])
    ax_lbp.bar(range(len(lbp_hist)), lbp_hist, color="steelblue", width=0.85, alpha=0.85)
    ax_lbp.set_title("LBP Histogram - Feature Extraction", fontsize=10)
    ax_lbp.set_xlabel("LBP Pattern Code", fontsize=9)
    ax_lbp.set_ylabel("Normalized Frequency", fontsize=9)
    ax_lbp.tick_params(labelsize=8)

    return fig


def build_analysis(img_bgr, fname="image"):
    """
    WRAPPER: Combines metrics extraction, visualization, and AI classification.
    For backward compatibility with existing code (UI.py, DIPCEP.py).
    
    Args:
        img_bgr: OpenCV image in BGR format
        fname: filename for title
        
    Returns:
        tuple: (matplotlib.figure.Figure, dict of metrics, dict of AI classification)
    """
    processed_data = extract_metrics(img_bgr)
    fig = create_visualization(processed_data, fname)
    metrics = processed_data["metrics"]
    ai_classification = classify_with_resnet50(img_bgr)
    return fig, metrics, ai_classification


# ============================================================================
# DATASET LOADING & CLASSIFICATION (NEW FEATURES FOR FULL PROJECT)
# ============================================================================

def load_dataset_labels(view_type="RANT"):
    """
    Load dataset labels from file.
    
    Args:
        view_type: "RANT" for anterior or "RPOST" for posterior
        
    Returns:
        dict: {filename: label (0/1)} - 0=normal, 1=metastasis
    """
    if view_type == "RANT":
        label_file = os.path.join(DATASET_PATH, "chestRANT", "chestRANT.txt")
    else:
        label_file = os.path.join(DATASET_PATH, "chestRPOST", "chestRPOST.txt")
    
    labels = {}
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    filename, label = parts
                    labels[filename] = int(label)
    return labels


def resolve_view_types(view_type):
    if isinstance(view_type, (list, tuple)):
        return [str(v).strip().upper() for v in view_type]

    view = str(view_type).strip().upper()
    if view in ("BOTH", "COMBINED", "ALL"):
        return ["RANT", "RPOST"]

    return [view]


def preprocess_for_resnet(img_bgr):
    """
    Preprocess image for ResNet50: convert to RGB, resize to 224x224, normalize.
    Use same preprocessing as training (no DIP filters to avoid domain shift).
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img_rgb)
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(img_pil).unsqueeze(0)


def classify_with_resnet50(img_bgr):
    """
    Classify image using ResNet50 model.
    Returns: {'prediction': 0 or 1, 'confidence': float, 'class': str}
    """
    resnet_model = load_resnet50_model()
    if resnet_model is None:
        return {"prediction": -1, "confidence": 0.0, "class": "Model not loaded"}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet_model.to(device)
    
    processed = preprocess_for_resnet(img_bgr).to(device)
    with torch.no_grad():
        output = resnet_model(processed).squeeze()
        probability = output.item()
    
    prediction = 1 if probability > 0.5 else 0
    confidence = probability if prediction == 1 else 1 - probability
    class_name = "Metastasis Detected" if prediction == 1 else "Healthy Bone"
    
    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "class": class_name,
        "probability": float(probability)
    }


def evaluate_classification_performance(predictions, ground_truth):
    """
    Calculate classification performance metrics.
    
    Args:
        predictions: list of predicted labels (0 or 1)
        ground_truth: list of actual labels (0 or 1)
        
    Returns:
        dict: performance metrics (sensitivity, specificity, accuracy, precision, F1)
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Confusion matrix
    tn = np.sum((predictions == 0) & (ground_truth == 0))  # True Negatives
    fp = np.sum((predictions == 1) & (ground_truth == 0))  # False Positives
    fn = np.sum((predictions == 0) & (ground_truth == 1))  # False Negatives
    tp = np.sum((predictions == 1) & (ground_truth == 1))  # True Positives
    
    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }


def batch_classify_dataset(view_type="RANT", limit=0):
    """
    Classify all images in dataset and evaluate performance.
    
    Args:
        view_type: "RANT", "RPOST", or "BOTH"/"COMBINED" for both views
        limit: 0 for all, or number to process
        
    Returns:
        dict: results with predictions, ground truth, and metrics
    """
    predictions = []
    ground_truth = []
    results_data = []

    count = 0
    for view in resolve_view_types(view_type):
        image_dir = os.path.join(DATASET_PATH, f"chest{view}")
        if not os.path.exists(image_dir):
            continue

        labels = load_dataset_labels(view)

        for filename in sorted(os.listdir(image_dir)):
            if limit and count >= limit:
                break

            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Get ground truth
            if filename not in labels:
                continue

            true_label = labels[filename]
            image_path = os.path.join(image_dir, filename)

            try:
                # Process image
                img_bgr = load_image_bgr(image_path)

                # Classify with ResNet50
                classification = classify_with_resnet50(img_bgr)

                pred = classification["prediction"]
                confidence = classification["confidence"]

                predictions.append(pred)
                ground_truth.append(true_label)

                results_data.append({
                    "filename": filename,
                    "true_label": true_label,
                    "predicted": pred,
                    "classification": classification["class"],
                    "confidence": confidence,
                    "probability": classification["probability"],
                    "view_type": view,
                })

                count += 1
            except Exception:
                continue
    
    # Calculate performance
    performance = evaluate_classification_performance(predictions, ground_truth)
    
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "results": results_data,
        "performance": performance,
        "total_images": count,
        "view_type": view_type,
        "view_types": resolve_view_types(view_type)
    }
