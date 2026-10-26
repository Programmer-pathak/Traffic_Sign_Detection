import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Traffic Sign Recognition AI System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern aesthetic UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .prediction-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)

# 43 Traffic Sign Categories
CLASSES_DICT = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection', 
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 
    16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution', 
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve', 
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 
    41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

# Sign Advice and Driving Instructions
SIGN_ADVICE = {
    0: "Maximum speed allowed is 20 km/h. School/residential zone.",
    1: "Maximum speed allowed is 30 km/h. Urban calm area.",
    2: "Maximum speed allowed is 50 km/h. Standard city road speed.",
    3: "Maximum speed allowed is 60 km/h. City ring road or boulevard.",
    4: "Maximum speed allowed is 70 km/h. Sub-urban arterial road.",
    5: "Maximum speed allowed is 80 km/h. Rural highway.",
    6: "Speed restriction of 80 km/h ends here. Resume normal road limit.",
    7: "Maximum speed allowed is 100 km/h. Dual carriageway or express road.",
    8: "Maximum speed allowed is 120 km/h. Motorway / Autobahn.",
    9: "Overtaking another moving vehicle is strictly prohibited.",
    10: "Heavy transport vehicles (>3.5T) cannot overtake.",
    11: "You have right of way at the immediate intersection ahead.",
    12: "Priority road: You have right of way on this main route.",
    13: "Yield / Give Way: Give way to oncoming and intersecting traffic.",
    14: "Full Stop mandatory: Bring vehicle to complete halt before crossing line.",
    15: "Road closed to all vehicles in both directions.",
    16: "Heavy goods vehicles over 3.5 metric tons not permitted.",
    17: "No Entry: Access forbidden for vehicles entering in this direction.",
    18: "General Caution: Potential hazard ahead, decrease speed.",
    19: "Sharp bend to the left ahead. Slow down before steering.",
    20: "Sharp bend to the right ahead. Slow down before steering.",
    21: "Series of bends or double curve ahead, first curve left/right.",
    22: "Uneven or bumpy road surface ahead. Drive carefully.",
    23: "Slippery road surface warning. Risk of skidding.",
    24: "Road narrows ahead on the right side. Merge safely.",
    25: "Road maintenance or construction workers on site ahead.",
    26: "Traffic light control ahead. Prepare to stop if red.",
    27: "Pedestrian crossing ahead. Be ready to yield to people.",
    28: "School or children crossing zone. Watch out for sudden movements.",
    29: "Bicycle crossing ahead. Watch out for cyclists.",
    30: "Risk of ice, frost, or snow on road surface. Reduce speed.",
    31: "Wildlife warning: Animals likely to cross the roadway.",
    32: "All specific speed and passing limits end. General road rules apply.",
    33: "Compulsory direction: Turn right ahead.",
    34: "Compulsory direction: Turn left ahead.",
    35: "Compulsory direction: Ahead only. No turns permitted.",
    36: "Compulsory direction: Go straight or make a right turn only.",
    37: "Compulsory direction: Go straight or make a left turn only.",
    38: "Pass the obstacle by keeping to the right.",
    39: "Pass the obstacle by keeping to the left.",
    40: "Roundabout ahead: Follow counter-clockwise circular motion.",
    41: "End of overtaking prohibition for standard vehicles.",
    42: "End of overtaking prohibition for trucks and heavy vehicles."
}

# Curated real-world road test samples
SAMPLE_SIGNS = {
    "sample_14_stop.png": {"name": "Stop Sign (Class 14)", "expected": 14},
    "sample_01_speed30.png": {"name": "Speed Limit 30 km/h (Class 1)", "expected": 1},
    "sample_13_yield.png": {"name": "Yield / Give Way (Class 13)", "expected": 13},
    "sample_17_no_entry.png": {"name": "No Entry (Class 17)", "expected": 17},
    "sample_11_right_of_way.png": {"name": "Right of Way at Intersection (Class 11)", "expected": 11},
    "sample_25_road_work.png": {"name": "Road Work (Class 25)", "expected": 25},
    "sample_28_children.png": {"name": "Children Crossing (Class 28)", "expected": 28},
    "sample_33_turn_right.png": {"name": "Turn Right Ahead (Class 33)", "expected": 33},
    "sample_38_keep_right.png": {"name": "Keep Right (Class 38)", "expected": 38},
    "sample_16_heavy_veh.png": {"name": "Vehicles > 3.5T Prohibited (Class 16)", "expected": 16}
}

@st.cache_resource
def load_trained_model(model_path="traffic_sign_model.keras"):
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(64, 64)):
    """
    Handles transparent background composites (PNG alpha channels),
    resizing to the model's input size, and normalizing RGB values.
    """
    if image.mode in ('RGBA', 'LA'):
        # Composite transparent pixels over neutral road/white background
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        img_rgb = bg
    else:
        img_rgb = image.convert("RGB")
        
    img_resized = img_rgb.resize(target_size)
    img_array = np.array(img_resized, dtype="float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
    st.markdown("### 🚦 System Controls")
    test_mode = st.radio(
        "Choose Mode:",
        ["🧪 Instant Real Road Samples", "📁 Upload Image", "📷 Live Camera", "📚 Sign Catalog"]
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Details")
    model = load_trained_model()
    if model is not None:
        st.success("✅ Model: `traffic_sign_model.keras`")
        st.caption(f"GTSRB classifier • {model.input_shape[1]}×{model.input_shape[2]} input")
    else:
        st.warning("⚠️ `traffic_sign_model.keras` not found. Run `python src/train.py` to train it.")

    st.markdown("---")
    st.caption("GTSRB: 43 German sign classes • Not a sign detector or safety system")

# Header
st.markdown('<div class="main-header">🚗 German Traffic Sign Recognition AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep CNN Computer Vision for Autonomous Road Sign Identification & Driver Safety Guidelines</div>', unsafe_allow_html=True)

# Main Container
if test_mode == "📚 Sign Catalog":
    st.subheader("📚 Complete GTSRB Traffic Sign Catalog (43 Classes)")
    cols = st.columns(3)
    for cid in range(43):
        col = cols[cid % 3]
        meta_img_path = os.path.join("data", "Meta", f"{cid}.png")
        with col:
            with st.container():
                st.markdown(f"**Class {cid}: {CLASSES_DICT[cid]}**")
                if os.path.exists(meta_img_path):
                    st.image(meta_img_path, width=80)
                st.caption(SIGN_ADVICE.get(cid, ""))
                st.markdown("---")

else:
    col1, col2 = st.columns([1, 1.2])

    input_image = None
    expected_class = None

    with col1:
        st.markdown("### 📥 Image Input")
        
        if test_mode == "🧪 Instant Real Road Samples":
            st.info("Pick any real GTSRB road test sign to verify prediction:")
            selected_sample_key = st.selectbox(
                "Choose Road Sample:",
                options=list(SAMPLE_SIGNS.keys()),
                format_func=lambda k: SAMPLE_SIGNS[k]["name"]
            )
            sample_info = SAMPLE_SIGNS[selected_sample_key]
            expected_class = sample_info["expected"]
            sample_img_path = os.path.join("data", "samples", selected_sample_key)
            if os.path.exists(sample_img_path):
                input_image = Image.open(sample_img_path)
                st.image(input_image, caption=f"Selected: {sample_info['name']}", width=220)
                st.caption(f"Ground Truth Class: **{expected_class} - {CLASSES_DICT[expected_class]}**")
            else:
                st.warning("Sample file not found.")

        elif test_mode == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a traffic sign image (.png, .jpg, .jpeg)",
                type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
                st.image(input_image, caption="Uploaded Image", use_container_width=True)
                st.caption("For best results, crop tightly around one sign; the model classifies signs but does not locate them in a full road scene.")

        elif test_mode == "📷 Live Camera":
            camera_file = st.camera_input("Capture sign using your webcam")
            if camera_file is not None:
                input_image = Image.open(camera_file)
                st.image(input_image, caption="Captured Image", use_container_width=True)
                st.caption("Keep one sign centered and large in the camera frame.")

    with col2:
        st.markdown("### 🧠 AI Prediction & Analysis")
        if input_image is not None:
            if model is None:
                st.error("Model weights `traffic_sign_model.h5` are not loaded.")
            else:
                with st.spinner("Analyzing image features through Deep CNN..."):
                    input_height, input_width = model.input_shape[1:3]
                    processed_img = preprocess_image(
                        input_image, target_size=(input_width, input_height)
                    )
                    preds = model.predict(processed_img, verbose=0)[0]

                    top_class = int(np.argmax(preds))
                    top_confidence = float(preds[top_class]) * 100

                    # Match indicator if ground truth is known
                    match_badge = ""
                    if expected_class is not None:
                        if top_class == expected_class:
                            match_badge = "✅ **100% Match with Ground Truth!**"
                        else:
                            match_badge = f"⚠️ Expected: {CLASSES_DICT[expected_class]}"

                    # Results Card
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="prediction-title">🎯 {CLASSES_DICT[top_class]}</div>
                        <p style="color: #475569; margin-top: 0.5rem; font-size: 1.05rem;">
                            <strong>Confidence:</strong> {top_confidence:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    if match_badge:
                        st.info(match_badge)

                    st.markdown(f"**Safety Rule & Driving Action:**\n> {SIGN_ADVICE.get(top_class, 'Follow standard road precautions.')}")

                    # Top-5 Predictions
                    st.markdown("#### 📊 Top-5 Prediction Probabilities")
                    top_5_idx = np.argsort(preds)[-5:][::-1]
                    for idx in top_5_idx:
                        cls_name = CLASSES_DICT[idx]
                        score = float(preds[idx]) * 100
                        st.write(f"**{cls_name}** ({score:.1f}%)")
                        st.progress(min(1.0, float(preds[idx])))
        else:
            st.info("👈 Please choose a sample sign or upload a photo on the left to see instant predictions.")