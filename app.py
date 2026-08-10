import os
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Optimize CPU and PyTorch threading for container environments / HF Spaces
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from model_def import DoodleCNN
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Set page configuration with premium dark theme aesthetics
st.set_page_config(
    page_title="Doodle Faithfulness: Grad-CAM vs HiResCAM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #f0f2f6;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .stAlert {
        border-radius: 8px;
    }
    .css-1e56321 {
        background-color: #1e293b;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Categories and Model
categories = ["airplane", "bicycle", "bus", "car", "train"]
device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = DoodleCNN(num_classes=len(categories)).to(device)
    # Checkpoint path in local assets
    checkpoint_path = os.path.join(os.path.dirname(__file__), "assets", "doodle_cnn.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

model = load_model()

# Cached attribution map generation
@st.cache_data
def generate_cam_map(method_name: str, target_class_idx: int, img_bytes: bytes):
    # Reconstruct input array from bytes
    img_np = np.frombuffer(img_bytes, dtype=np.float32).reshape(1, 1, 28, 28)
    input_tensor = torch.tensor(img_np).to(device)
    targets = [ClassifierOutputTarget(target_class_idx)]
    target_layers = [model.conv2]
    
    if method_name == "Grad-CAM":
        cam_extractor = GradCAM(model=model, target_layers=target_layers)
    else:
        cam_extractor = HiResCAM(model=model, target_layers=target_layers)
        
    cam_map = cam_extractor(input_tensor=input_tensor, targets=targets)[0]
    return cam_map

# 2. Setup robust normalization function
def normalize_heatmap(heatmap):
    hmin = heatmap.min()
    hmax = heatmap.max()
    if hmax - hmin < 1e-6:
        return np.zeros_like(heatmap)
    # Contrast stretch to [0, 1]
    normalized = (heatmap - hmin) / (hmax - hmin)
    # Apply power transform to enhance contrast (sqrt to spread out low values)
    normalized = np.power(normalized, 0.5)
    return np.clip(normalized, 0.0, 1.0)

# Sidebar configurations
st.sidebar.title("Configuration")
st.sidebar.markdown("Configure explanation settings and visualization options below.")

# Option to select sample or upload
input_mode = st.sidebar.radio("Select Input Source:", ["Preset Sample Doodles", "Upload Custom Image"])

# Explanation options
methods_to_show = st.sidebar.multiselect(
    "Select Attribution Methods:",
    ["Grad-CAM", "HiResCAM"],
    default=["Grad-CAM", "HiResCAM"]
)

attribution_view = st.sidebar.radio(
    "Attribution View Mode:",
    ["Binary Stroke Masking", "Raw Attribution"],
    index=0,
    help="Select whether to restrict attribution to drawn strokes or show raw, unmasked activations."
)
use_stroke_masking = (attribution_view == "Binary Stroke Masking")

# Main title
st.title("Doodle Explainability Faithfulness Demo")
st.markdown("### Comparing **Grad-CAM** vs. **HiResCAM** on Doodle Classification tasks.")

with st.expander(" Quick Start & How to Use Guide", expanded=False):
    st.markdown("""
    Welcome to the **Doodle Explainability Faithfulness Demo**! This tool helps you understand how neural networks classify drawings and visualizes where the model is looking when making a decision.
    
    ###  Step-by-Step Guide
    1. **Choose an Input Source (in the sidebar):**
       * **Preset Sample Doodles:** Choose one of our pre-loaded doodle samples.
       * **Upload Custom Image:** Upload your own PNG/JPG drawing.
    2. **View the Prediction (Left Column):**
       * The model outputs its predicted class and confidence.
       * The bar chart shows how likely the model thinks each class is.
    3. **Explore Visual Explanations (Right Column):**
       * The heatmaps highlight the regions of the drawing the model focused on. **Warm colors (yellow/red)** mean high importance.
       * **Grad-CAM** and **HiResCAM** are two different explainability algorithms. In our experiments on this dataset, HiResCAM demonstrated stronger faithfulness to the model's actual features.
    4. **Try Interacting:**
       * **Select target class to explain:** Use the dropdown below the explanations to see what parts of the doodle look like *other* classes to the model (the underlying doodle remains the same, but the heatmap updates!).
       * **Attribution View Mode (in the sidebar):** Toggle to 'Raw Attribution' to see the model's full spatial response.
    """)

st.markdown("---")

# Load samples
assets_dir = os.path.join(os.path.dirname(__file__), "assets")
sample_images_path = os.path.join(assets_dir, "sample_images.npy")
sample_labels_path = os.path.join(assets_dir, "sample_labels.npy")

# Input Image Handling
img_to_run = None
true_label_idx = None

if input_mode == "Preset Sample Doodles":
    if os.path.exists(sample_images_path) and os.path.exists(sample_labels_path):
        sample_images = np.load(sample_images_path)
        sample_labels = np.load(sample_labels_path)
        
        st.sidebar.subheader("Select Preset Doodle")
        sample_idx = st.sidebar.selectbox(
            "Choose a sample image:",
            range(len(sample_images)),
            format_func=lambda idx: f"Sample #{idx} (True Class: {categories[sample_labels[idx]].title()})"
        )
        
        img_to_run = sample_images[sample_idx]  # Shape: (1, 28, 28)
        true_label_idx = int(sample_labels[sample_idx])
    else:
        st.error("Sample images/labels not found in `demo/assets/` folder.")
else:
    st.sidebar.subheader("Upload Custom Doodle")
    uploaded_file = st.sidebar.file_uploader("Upload PNG or JPG image (processed to 28x28):", type=["png", "jpg", "jpeg"])
    invert_colors = st.sidebar.checkbox("Invert Colors (Check if background is light and strokes are dark)", value=True)
    
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("L")
        pil_img = pil_img.resize((28, 28))
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        if invert_colors:
            img_np = 1.0 - img_np
        img_to_run = np.expand_dims(img_np, axis=0)  # Shape: (1, 28, 28)
    else:
        st.info("Please upload an image or switch to preset samples in the sidebar.")
 
if img_to_run is not None:
    # Run Inference
    input_tensor = torch.tensor(img_to_run).unsqueeze(0).to(device)  # shape: (1, 1, 28, 28)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0).numpy()
        pred_idx = np.argmax(probs)
        pred_confidence = probs[pred_idx]
        
    # Main columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Prediction")
        
        # Display nicely formatted classification prediction
        st.metric(label="Predicted Class", value=categories[pred_idx].title())
        st.metric(label="Confidence", value=f"{pred_confidence * 100:.2f}%")
        
        if true_label_idx is not None:
            st.markdown(f"**True Label:** `{categories[true_label_idx].title()}`")
            if pred_idx == true_label_idx:
                st.success("Correct Prediction!")
            else:
                st.warning("Incorrect Prediction!")
                st.info(" **Debugging Tip:** Even on incorrect predictions, attribution methods can reveal why the model was confused — useful for debugging model behavior, not just validating correct decisions.")
                
        if pred_confidence < 0.65:
            st.info(" **NOTE:** Model confidence is low for this input. Heatmaps may be diffuse and less interpretable, since the model has weak evidence for any single class — try a clearer doodle for more meaningful explanations.")
        
        # Plot all class probabilities
        fig_bar, ax_bar = plt.subplots(figsize=(4, 3))
        y_pos = np.arange(len(categories))
        ax_bar.barh(y_pos, probs, color='#3b82f6', edgecolor='#1d4ed8')
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([c.title() for c in categories])
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel('Probability')
        ax_bar.set_xlim(0, 1.05)
        fig_bar.patch.set_facecolor('#0f1116')
        ax_bar.set_facecolor('#1e293b')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['left'].set_color('#94a3b8')
        ax_bar.spines['bottom'].set_color('#94a3b8')
        ax_bar.tick_params(colors='#94a3b8')
        ax_bar.xaxis.label.set_color('#94a3b8')
        plt.tight_layout()
        st.pyplot(fig_bar)
        
    with col2:
        st.subheader("Visual Explanations")
        
        # Prepare targets
        target_class_idx = st.selectbox(
            "Select target class to explain:",
            range(len(categories)),
            index=int(pred_idx),
            format_func=lambda idx: categories[idx].title()
        )
        
        if target_class_idx != pred_idx and use_stroke_masking:
            if "info_dismissed" not in st.session_state:
                st.session_state.info_dismissed = False
            
            if not st.session_state.info_dismissed:
                col_text, col_btn = st.columns([12, 1])
                with col_text:
                    st.info(" **NOTE:** Because stroke masking restricts attribution to the doodle's drawn pixels, changing the target class here will only change heat intensity on existing strokes, not reveal new regions — switch to 'Raw attribution' view to see the full, unmasked spatial difference between classes.")
                with col_btn:
                    if st.button("✕", key="dismiss_info", help="Dismiss this tip"):
                        st.session_state.info_dismissed = True
                        st.rerun()
        else:
            st.session_state.info_dismissed = False
        
        targets = [ClassifierOutputTarget(target_class_idx)]
        
        # Compute CAMs
        target_layers = [model.conv2]
        
        raw_img = img_to_run[0]  # shape: (28, 28)
        stroke_mask = (raw_img > 0.0).astype(np.float32)
        
        cols_viz = st.columns(1 + len(methods_to_show))
        
        # Original Image
        with cols_viz[0]:
            st.write("**Original Doodle**")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(raw_img, cmap="gray")
            ax.axis("off")
            fig.patch.set_facecolor('#0f1116')
            st.pyplot(fig)
            
        # Explanations
        img_bytes = img_to_run.astype(np.float32).tobytes()
        cam_cache = {}
        
        for i, method_name in enumerate(methods_to_show):
            with cols_viz[i + 1]:
                st.write(f"**{method_name}**")
                
                # Fetch cached CAM map
                cam_map = generate_cam_map(method_name, target_class_idx, img_bytes)
                cam_cache[method_name] = cam_map
                
                # Calculate value range to check for near-blank maps
                cam_range = float(cam_map.max() - cam_map.min())
                
                # Apply stroke masking if enabled
                if use_stroke_masking:
                    cam_map = cam_map * stroke_mask
                    
                # Normalize
                cam_normalized = normalize_heatmap(cam_map)
                
                # Plot
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(raw_img, cmap="gray")
                ax.imshow(cam_normalized, cmap="inferno", alpha=0.5, vmin=0.0, vmax=1.0)
                ax.axis("off")
                fig.patch.set_facecolor('#0f1116')
                st.pyplot(fig)
                
                # Check for near-blank maps
                if cam_range < 1e-4:
                    if method_name == "Grad-CAM":
                        st.info("**WARNING: **Grad-CAM** produced a near-blank map for this input/class combination. This is a known limitation — Grad-CAM's final ReLU step clamps negative gradient signals to zero, which can occur when the model finds weak or no positive evidence for this class. In our paper's evaluation set, this occurred in 12 of 100 candidate images, compared to 0 for HiResCAM (see Section 3.6).")
                    else:
                        st.info(f"**WARNING:** **{method_name}** produced a near-blank map for this input/class combination. This is a known limitation — explainers using ReLU or gradient clamping can return empty maps when the model finds weak or no positive evidence for this class. In our paper's evaluation set, this occurred in 0 of 100 candidate images for HiResCAM, compared to 12 for Grad-CAM (see Section 3.6).")
                
        # Compare metrics or differences
        if "Grad-CAM" in methods_to_show and "HiResCAM" in methods_to_show:
            gc_map = cam_cache.get("Grad-CAM", generate_cam_map("Grad-CAM", target_class_idx, img_bytes))
            hc_map = cam_cache.get("HiResCAM", generate_cam_map("HiResCAM", target_class_idx, img_bytes))
            
            if use_stroke_masking:
                gc_map = gc_map * stroke_mask
                hc_map = hc_map * stroke_mask
                
            gc_norm = normalize_heatmap(gc_map)
            hc_norm = normalize_heatmap(hc_map)
            
            mae_diff = np.mean(np.abs(gc_norm - hc_norm))
            st.markdown(f"**Mean Absolute Difference between Grad-CAM & HiResCAM:** `{mae_diff:.4f}`")
            st.caption("*(Supplementary exploratory metric of heatmap dissimilarity, not part of the core faithfulness evaluation.)*")
            
            # Simple explanation note
            st.info("NOTE: In our experiments on sparse doodle data, HiResCAM's element-wise gradient formulation showed stronger faithfulness metrics than Grad-CAM's spatial averaging (see accompanying paper for full quantitative results).")
