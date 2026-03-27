"""
Oakeshott Sword Classifier — Inference & Streamlit Demo
--------------------------------------------------------
Hierarchical inference pipeline:
  1. Coarse model predicts main sword type
  2. If a fine model exists for that type, predicts subtype
  3. Grad-CAM heatmap shows which image regions influenced the prediction
  4. Streamlit UI displays results with historical context

Requirements:
    pip install torch torchvision streamlit grad-cam Pillow numpy matplotlib

Usage:
    streamlit run inference.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import models, transforms
from pathlib import Path
from PIL import Image, ImageOps
import streamlit as st
import io
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG — update paths if needed
# ---------------------------------------------------------------------------

MODELS_DIR = Path(r"D:\oakeshott_models")
IMG_SIZE   = 224

# Confidence threshold below which subtype prediction is suppressed
# and only main type is returned
SUBTYPE_CONFIDENCE_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# HISTORICAL CONTEXT
# Each type has: date range, description, key features
# ---------------------------------------------------------------------------

TYPE_INFO = {
    "Type_X": {
        "period": "9th – 12th century",
        "description": "One of the earliest true knightly swords, used from the Viking Age through the early Crusades. Broad, flat blade with a wide and shallow fuller running most of the blade length.",
        "features": "Wide shallow fuller, broad blade, brazil-nut or disk pommel, straight quillons"
    },
    "Type_Xa": {
        "period": "10th – 12th century",
        "description": "A longer, more slender variant of Type X favoured for mounted combat. Retains the wide fuller but with a more tapered profile toward the point.",
        "features": "Longer blade, narrower taper, wide fuller, suitable for mounted use"
    },
    "Type_XI": {
        "period": "11th – 13th century",
        "description": "The classic Crusader-era sword. Slender blade with a fuller that stops short of the tip, giving a more acute point for early mail-piercing use.",
        "features": "Slender blade, fuller ending before tip, brazil-nut pommel, long quillons"
    },
    "Type_XII": {
        "period": "12th – 14th century",
        "description": "A versatile knightly sword balancing cut and thrust. Fuller runs approximately half the blade length, leaving a stiff lower half.",
        "features": "Half-length fuller, balanced taper, wheel or disk pommel"
    },
    "Type_XIIa": {
        "period": "12th – 14th century",
        "description": "The great sword variant of Type XII, requiring two hands. Longer grip and blade suited for the battlefield rather than personal defense.",
        "features": "Extended grip, longer blade, half-length fuller, two-handed use"
    },
    "Type_XIII": {
        "period": "13th – 14th century",
        "description": "The classic war sword of the high medieval period. Wide blade, flat cross-section, and a wide fuller designed for powerful cutting strokes.",
        "features": "Wide blade, flat cross-section, broad fuller, spatulate tip"
    },
    "Type_XIIIa": {
        "period": "13th – 14th century",
        "description": "The great sword of war — a two-handed version of Type XIII used by infantry in mass battles. One of the largest swords of the medieval period.",
        "features": "Very long blade, two-handed grip, wide fuller, broad cutting blade"
    },
    "Type_XIV": {
        "period": "13th – 15th century",
        "description": "A compact, single-handed sword designed for close personal combat. Shorter blade with a pronounced taper and often curved quillons.",
        "features": "Short blade, strong taper, curved quillons, compact design"
    },
    "Type_XV": {
        "period": "14th – 15th century",
        "description": "The quintessential late-medieval thrusting sword, developed in response to improving plate armour. Stiff blade with a diamond cross-section and acute point.",
        "features": "Diamond cross-section, acute point, no fuller, stiff blade"
    },
    "Type_XVa": {
        "period": "14th – 15th century",
        "description": "A longer hand-and-a-half variant of Type XV. Extended grip allows two-handed use while maintaining the thrusting-optimised blade geometry.",
        "features": "Extended grip, longer blade, diamond cross-section, acute point"
    },
    "Type_XVI": {
        "period": "14th – 15th century",
        "description": "A transitional type combining a short fuller at the forte with a stiff lower half, balancing earlier cutting traditions with the new thrusting demands.",
        "features": "Short forte fuller, stiff lower blade, hexagonal or diamond cross-section"
    },
    "Type_XVIa": {
        "period": "14th – 15th century",
        "description": "Hand-and-a-half version of Type XVI, popular among men-at-arms who needed versatility against both plate and mail.",
        "features": "Extended grip, short forte fuller, versatile cut-and-thrust geometry"
    },
    "Type_XVII": {
        "period": "14th – 15th century",
        "description": "A specialised armour-piercing sword with an extremely stiff, flattened hexagonal blade designed to find gaps in plate armour.",
        "features": "Hexagonal cross-section, very stiff blade, narrow profile, armour-piercing"
    },
    "Type_XVIII": {
        "period": "15th century",
        "description": "The elegant sword of the high Gothic period. Graceful blade with a gentle taper, often with a short forte fuller, balancing beauty and function.",
        "features": "Gentle taper, short forte fuller or none, elegant profile, wheel pommel"
    },
    "Type_XVIIIa": {
        "period": "15th century",
        "description": "Hand-and-a-half variant of Type XVIII, popular among German and Italian men-at-arms. Combines elegant proportions with extended reach.",
        "features": "Extended grip, gentle taper, hand-and-a-half use, Gothic aesthetic"
    },
    "Type_XVIIIb": {
        "period": "15th century",
        "description": "Two-handed variant of the Type XVIII family, the great sword of the late Gothic period. Requires both hands and features a longer ricasso.",
        "features": "Two-handed, longer blade, ricasso, Gothic cross-guard"
    },
    "Type_XVIIIc": {
        "period": "15th century",
        "description": "A robust variant of Type XVIII with a broader, more substantial blade. Associated with German production and Burgundian military fashion.",
        "features": "Broader blade, robust profile, short forte fuller"
    },
    "Type_XVIIIe": {
        "period": "15th century",
        "description": "The Danish variant of the Type XVIII family, characterised by a flat wide blade with a pronounced central ridge. Distinctive Scandinavian morphology.",
        "features": "Wide flat blade, central ridge, pronounced midrib, Danish origin"
    },
    "Type_XIX": {
        "period": "15th – early 16th century",
        "description": "A slender, late-medieval sword with a hexagonal or flattened blade and minimal or no fuller. Transitional between the medieval and Renaissance periods.",
        "features": "Hexagonal blade, minimal fuller, slender profile, late medieval"
    },
    "Type_XX": {
        "period": "15th – 16th century",
        "description": "A broad-bladed late medieval sword retaining earlier cutting traditions. Often associated with civilian use and the transition toward the Renaissance sword.",
        "features": "Broad blade, flat cross-section, wide fuller, civilian use"
    },
    "Type_XXa": {
        "period": "15th – 16th century",
        "description": "Variant of Type XX with a more pronounced taper and occasionally a secondary fuller. Bridges the gap between war sword and civilian side-sword.",
        "features": "Tapered blade, secondary fuller possible, transitional form"
    },
    "Type_XXI": {
        "period": "15th – 16th century",
        "description": "One of the final forms in the Oakeshott typology, transitioning toward the Renaissance side-sword. Complex hilts begin to appear alongside the medieval blade form.",
        "features": "Transitional hilt elements, late medieval blade, precursor to side-sword"
    },
}

# Maps coarse class names to fine model folder names
# Only types that have a working fine model are listed here
FINE_MODEL_MAP = {
    "Type_X":     "fine_Type_X",
    "Type_XIII":  "fine_Type_XIII",
    "Type_XV":    "fine_Type_XV",
    "Type_XVI":   "fine_Type_XVI",
    "Type_XVIII": "fine_Type_XVIII",
    "Type_XX":    "fine_Type_XX",
}

# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

class ResizeWithPadding:
    def __init__(self, size=IMG_SIZE):
        self.size = size

    def __call__(self, img):
        img.thumbnail((self.size, self.size), Image.LANCZOS)
        return ImageOps.pad(img, (self.size, self.size), color=(128, 128, 128))


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

inference_transforms = transforms.Compose([
    ResizeWithPadding(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: Path):
    """Load a trained model and its class map from a directory."""
    checkpoint = torch.load(
        model_dir / "best_model.pth",
        map_location=device
    )
    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)

    return model, classes


@st.cache_resource
def load_all_models():
    """Load coarse model and all available fine models. Cached by Streamlit."""
    models_dict = {}

    # Coarse model
    coarse_dir = MODELS_DIR / "coarse"
    if coarse_dir.exists():
        models_dict["coarse"] = load_model(coarse_dir)
        print("Loaded coarse model")

    # Fine models
    for type_name, folder_name in FINE_MODEL_MAP.items():
        fine_dir = MODELS_DIR / folder_name
        if fine_dir.exists():
            models_dict[type_name] = load_model(fine_dir)
            print(f"Loaded fine model: {folder_name}")

    return models_dict

# ---------------------------------------------------------------------------
# GRAD-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlights regions the model focused on when making its prediction.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into the last conv layer of ResNet50 (layer4)
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a given class index.
        Returns a numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Backpropagate for the target class
        score = output[0, class_idx]
        score.backward()

        # Weight activations by gradient importance
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Resize to input image size
        cam = torch.nn.functional.interpolate(
            cam, size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


def overlay_gradcam(original_img: Image.Image,
                    heatmap: np.ndarray,
                    alpha: float = 0.45) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on the original image.
    Returns a PIL Image.
    """
    # Resize original to match heatmap
    img_resized = original_img.resize((IMG_SIZE, IMG_SIZE))
    img_array   = np.array(img_resized.convert("RGB")) / 255.0

    # Apply colormap to heatmap
    colormap    = cm.get_cmap("jet")
    heatmap_rgb = colormap(heatmap)[:, :, :3]

    # Blend
    blended = (1 - alpha) * img_array + alpha * heatmap_rgb
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)

# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def predict(image: Image.Image,
            all_models: dict) -> dict:
    """
    Returns a dict with:
        main_type       : predicted main type string
        main_confidence : confidence for main type (0-1)
        top3            : list of (type_name, confidence) top 3 coarse predictions
        subtype         : predicted subtype string or None
        sub_confidence  : subtype confidence or None
        sub_top3        : subtype top 3 or None
        gradcam_coarse  : PIL Image with coarse Grad-CAM overlay
        gradcam_fine    : PIL Image with fine Grad-CAM overlay or None
    """
    if "coarse" not in all_models:
        return {"error": "Coarse model not found"}

    coarse_model, coarse_classes = all_models["coarse"]

    # Prepare input tensor
    tensor = inference_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    # --- Coarse prediction ---
    with torch.no_grad():
        coarse_logits = coarse_model(tensor)
        coarse_probs  = torch.softmax(coarse_logits, dim=1)[0]

    # Re-run with gradients for Grad-CAM
    gradcam_coarse_gen = GradCAM(coarse_model)
    tensor_grad = inference_transforms(
        image.convert("RGB")
    ).unsqueeze(0).to(device)
    tensor_grad.requires_grad_(True)
    logits_for_cam = coarse_model(tensor_grad)
    probs_for_cam  = torch.softmax(logits_for_cam, dim=1)[0]

    main_idx        = probs_for_cam.argmax().item()
    main_type       = coarse_classes[main_idx]
    main_confidence = probs_for_cam[main_idx].item()

    # Coarse Grad-CAM
    cam_coarse    = gradcam_coarse_gen.generate(tensor_grad, main_idx)
    gradcam_coarse_img = overlay_gradcam(image, cam_coarse)

    # Top 3 coarse predictions
    top3_indices = probs_for_cam.topk(min(3, len(coarse_classes))).indices
    top3 = [
        (coarse_classes[i.item()], probs_for_cam[i.item()].item())
        for i in top3_indices
    ]

    result = {
        "main_type":       main_type,
        "main_confidence": main_confidence,
        "top3":            top3,
        "subtype":         None,
        "sub_confidence":  None,
        "sub_top3":        None,
        "gradcam_coarse":  gradcam_coarse_img,
        "gradcam_fine":    None,
    }

    # --- Fine prediction ---
    if main_type in all_models and main_confidence >= SUBTYPE_CONFIDENCE_THRESHOLD:
        fine_model, fine_classes = all_models[main_type]

        gradcam_fine_gen = GradCAM(fine_model)
        tensor_fine = inference_transforms(
            image.convert("RGB")
        ).unsqueeze(0).to(device)
        tensor_fine.requires_grad_(True)

        fine_logits = fine_model(tensor_fine)
        fine_probs  = torch.softmax(fine_logits, dim=1)[0]

        sub_idx        = fine_probs.argmax().item()
        subtype        = fine_classes[sub_idx]
        sub_confidence = fine_probs[sub_idx].item()

        cam_fine       = gradcam_fine_gen.generate(tensor_fine, sub_idx)
        gradcam_fine_img = overlay_gradcam(image, cam_fine)

        sub_top3_indices = fine_probs.topk(min(3, len(fine_classes))).indices
        sub_top3 = [
            (fine_classes[i.item()], fine_probs[i.item()].item())
            for i in sub_top3_indices
        ]

        result["subtype"]       = subtype
        result["sub_confidence"] = sub_confidence
        result["sub_top3"]      = sub_top3
        result["gradcam_fine"]  = gradcam_fine_img

    return result

# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------

def format_type_name(type_str: str) -> str:
    """Convert Type_XVIIIa to 'Type XVIIIa' for display."""
    return type_str.replace("_", " ")


def confidence_color(conf: float) -> str:
    if conf >= 0.7:
        return "#2ecc71"   # green
    elif conf >= 0.5:
        return "#f39c12"   # orange
    else:
        return "#e74c3c"   # red


def render_confidence_bar(label: str, confidence: float):
    color = confidence_color(confidence)
    pct   = int(confidence * 100)
    st.markdown(f"""
    <div style="margin-bottom:6px">
        <div style="display:flex; justify-content:space-between; margin-bottom:2px">
            <span style="font-size:14px">{label}</span>
            <span style="font-size:14px; font-weight:bold; color:{color}">{pct}%</span>
        </div>
        <div style="background:#e0e0e0; border-radius:4px; height:10px">
            <div style="background:{color}; width:{pct}%; height:10px; border-radius:4px"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Oakeshott Sword Classifier",
        page_icon="⚔️",
        layout="wide"
    )

    # Header
    st.title("⚔️ Oakeshott Sword Typology Classifier")
    st.markdown(
        "Upload a photograph of a medieval sword to identify its "
        "[Oakeshott typology](https://oakeshott.org/) classification."
    )
    st.divider()

    # Load models
    with st.spinner("Loading models..."):
        all_models = load_all_models()

    if "coarse" not in all_models:
        st.error(
            f"Could not find coarse model at `{MODELS_DIR / 'coarse'}`. "
            "Please check the MODELS_DIR path in the config."
        )
        return

    st.success(
        f"Loaded coarse model + "
        f"{len(all_models) - 1} fine models"
    )

    # Upload
    uploaded = st.file_uploader(
        "Upload a sword image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded is None:
        st.info("Upload an image above to begin classification.")
        return

    image = Image.open(uploaded).convert("RGB")

    # Run inference
    with st.spinner("Analysing..."):
        result = predict(image, all_models)

    if "error" in result:
        st.error(result["error"])
        return

    # -----------------------------------------------------------------------
    # RESULTS LAYOUT
    # -----------------------------------------------------------------------

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    with col_results:
        main_type = result["main_type"]
        main_conf = result["main_confidence"]
        subtype   = result["subtype"]
        sub_conf  = result["sub_confidence"]

        # Primary prediction
        st.subheader("Classification Result")

        if subtype and sub_conf and sub_conf >= SUBTYPE_CONFIDENCE_THRESHOLD:
            display_type = subtype
            display_conf = sub_conf
        else:
            display_type = main_type
            display_conf = main_conf

        color = confidence_color(display_conf)
        st.markdown(
            f"<h2 style='color:{color}'>{format_type_name(display_type)}</h2>",
            unsafe_allow_html=True
        )

        # Confidence warning
        if display_conf < 0.5:
            st.warning(
                "⚠️ Low confidence — the model is uncertain. "
                "Consider the top 3 alternatives below."
            )
        elif display_conf < 0.7:
            st.info("ℹ️ Moderate confidence prediction.")

        # Main type confidence bar
        st.markdown("**Main type confidence:**")
        render_confidence_bar(format_type_name(main_type), main_conf)

        # Subtype confidence bar if available
        if subtype and sub_conf:
            st.markdown("**Subtype confidence:**")
            render_confidence_bar(format_type_name(subtype), sub_conf)

        # Top 3 alternatives
        st.markdown("**Top 3 main type candidates:**")
        for type_name, conf in result["top3"]:
            render_confidence_bar(format_type_name(type_name), conf)

        if result["sub_top3"] and len(result["sub_top3"]) > 1:
            st.markdown("**Top 3 subtype candidates:**")
            for type_name, conf in result["sub_top3"]:
                render_confidence_bar(format_type_name(type_name), conf)

    st.divider()

    # -----------------------------------------------------------------------
    # HISTORICAL CONTEXT
    # -----------------------------------------------------------------------

    info_key = subtype if (subtype and sub_conf and
                           sub_conf >= SUBTYPE_CONFIDENCE_THRESHOLD) else main_type

    if info_key in TYPE_INFO:
        info = TYPE_INFO[info_key]
        st.subheader(f"📜 Historical Context — {format_type_name(info_key)}")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Period", info["period"])
        with col_b:
            st.markdown(f"**Description**\n\n{info['description']}")
        with col_c:
            st.markdown(f"**Key Features**\n\n{info['features']}")

    st.divider()

    # -----------------------------------------------------------------------
    # GRAD-CAM VISUALIZATIONS
    # -----------------------------------------------------------------------

    st.subheader("🔍 Model Attention — Grad-CAM")
    st.markdown(
        "These heatmaps show which regions of the image most influenced "
        "the model's decision. Red/yellow areas had the highest impact."
    )

    cam_cols = [result["gradcam_coarse"]]
    cam_labels = [f"Coarse: {format_type_name(main_type)} ({main_conf*100:.0f}%)"]

    if result["gradcam_fine"]:
        cam_cols.append(result["gradcam_fine"])
        cam_labels.append(
            f"Fine: {format_type_name(subtype)} ({sub_conf*100:.0f}%)"
        )

    cols = st.columns(len(cam_cols))
    for col, img, label in zip(cols, cam_cols, cam_labels):
        with col:
            st.image(img, caption=label, use_column_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    # SIDE BY SIDE: uploaded vs padded input
    # -----------------------------------------------------------------------

    with st.expander("Show preprocessed input fed to model"):
        padded = ResizeWithPadding(IMG_SIZE)(image.convert("RGB"))
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            st.image(padded, caption=f"Resized & padded ({IMG_SIZE}×{IMG_SIZE})",
                     use_column_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<small>Oakeshott Typology Classifier · ResNet50 · "
        "Hierarchical coarse-to-fine classification</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()