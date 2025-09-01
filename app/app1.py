import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from datetime import datetime
from fpdf import FPDF

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="KrishiRakshak | Plant Disease Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR A MODERN UI ---
st.markdown("""
<style>
    /* General Body Styling */
    body {
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Main App Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Title Styling */
    .title {
        font-size: 3.5rem;
        font-weight: bold;
        color: #1A535C; /* Dark Teal */
        text-align: center;
        padding-bottom: 1rem;
    }
    /* Header/Subheader Styling */
    h1, h2, h3 {
        color: #2a7a85; /* Softer Teal */
    }
    /* Result Text Styling */
    .result-text {
        font-size: 2rem;
        font-weight: bold;
        color: #F7B801; /* Saffron Yellow for contrast */
        text-align: center;
        padding: 1rem;
        border: 2px dashed #F7B801;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    /* Custom Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        border: 2px solid #1A535C;
        background-color: #1A535C;
        color: white;
        height: 3.5em;
        font-size: 1.1em;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: white;
        color: #1A535C;
        border-color: #1A535C;
    }
    .stButton>button:active {
        transform: scale(0.98);
    }
    /* Download Button Styling */
    .stDownloadButton>button {
        background-color: #FF6B6B; /* Coral Red */
        color: white;
        border-radius: 25px;
        border: 2px solid #FF6B6B;
        height: 3em;
        width: 100%;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stDownloadButton>button:hover {
        background-color: white;
        color: #FF6B6B;
    }
    /* File Uploader Styling */
    .stFileUploader {
        border: 2px dashed #2a7a85;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #fafafa;
    }
    /* Expander styling */
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- TREATMENT RECOMMENDATIONS DATABASE ---
treatment_recommendations = {
    "Apple___Apple_scab": {
        "overview": "Apple Scab is a fungal disease that causes olive-green to brown spots on leaves, fruit, and twigs.",
        "treatment": "Apply fungicides like myclobutanil or captan, starting from the green tip stage. Prune trees to improve air circulation and sunlight penetration. Rake and destroy fallen leaves in autumn to reduce spore survival.",
        "prevention": "Plant resistant varieties like Liberty, Enterprise, or Pristine. Ensure proper spacing and apply a preventative fungicide spray program in early spring.",
    },
    "Apple___Black_rot": {
        "overview": "Black Rot is a fungal disease causing leaf spots, cankers on branches, and a firm, black rot on the fruit.",
        "treatment": "Prune out and destroy cankered branches during dormancy. Apply fungicides, such as captan or sulfur-based sprays, from bud break through the growing season. Remove mummified fruit from the trees.",
        "prevention": "Maintain good sanitation by removing dead wood and fallen fruit. Ensure proper air circulation through pruning. Control insects that can create wounds for the fungus to enter.",
    },
    "Peach___Bacterial_spot": {
        "overview": "Bacterial Spot is a disease caused by bacteria, leading to purple-black spots on leaves and pitted spots on fruit.",
        "treatment": "Apply copper-based bactericides during dormancy and early in the season. Oxytetracycline sprays can be used during the growing season. Prune to improve air circulation and promote rapid drying of foliage.",
        "prevention": "Plant resistant peach varieties if available. Avoid overhead irrigation to keep foliage dry. Maintain tree vigor with proper fertilization and watering, as stressed trees are more susceptible.",
    },
    "Tomato___Late_blight": {
        "overview": "Late Blight is a destructive fungal disease that causes large, water-soaked, greenish-black lesions on leaves and fruit.",
        "treatment": "Apply fungicides containing chlorothalonil, mancozeb, or copper as soon as symptoms appear. Remove and destroy infected plants immediately to prevent spread. Ensure thorough coverage of all plant surfaces.",
        "prevention": "Plant certified disease-free seeds and plants. Allow for ample spacing between plants for good air circulation. Avoid overhead watering, especially late in the day. Rotate crops and remove volunteer tomato and potato plants.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "overview": "Northern Leaf Blight is a fungal disease characterized by long, elliptical, grayish-green or tan lesions on corn leaves.",
        "treatment": "Fungicide applications (strobilurin or triazole-based) can be effective if applied early, especially on susceptible hybrids in high-risk conditions.",
        "prevention": "The most effective method is planting resistant corn hybrids. Tillage to bury crop residue can reduce fungal survival. Crop rotation away from corn for at least one year is also beneficial.",
    },
    # Generic advice for healthy plants
    "default_healthy": {
        "overview": "The plant appears to be in good health!",
        "treatment": "No treatment is necessary. Continue your current care routine.",
        "prevention": "To maintain plant health, ensure consistent watering, provide adequate sunlight, and use a balanced fertilizer according to the plant's needs. Monitor regularly for any signs of pests or disease.",
    },
    # Default message for diseases not in this dictionary
    "default_disease": {
        "overview": "Disease detected. Specific recommendations are not yet available in our database.",
        "treatment": "Isolate the affected plant to prevent potential spread. Remove and dispose of heavily affected leaves or branches.",
        "prevention": "Research the specific disease online or consult a local agricultural extension office for tailored advice. Provide good air circulation and avoid over-watering.",
    }
}


# --- PDF GENERATION FUNCTION ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'KrishiRakshak - Plant Disease Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_report(image, disease_name, confidence_score, recommendations):
    pdf = PDF()
    pdf.add_page()

    # --- REPORT METADATA ---
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 10, 'Diagnosis Report', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Report Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
    pdf.ln(10)

    # --- IMAGE AND DIAGNOSIS ---
    temp_image_path = "temp_uploaded_leaf.png"
    image.save(temp_image_path)
    pdf.image(temp_image_path, x=15, y=pdf.get_y(), w=70)
    pdf.set_xy(100, pdf.get_y() + 20)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Detected Disease:', 0, 1, 'L')
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(220, 50, 50)
    pdf.multi_cell(0, 10, f"{disease_name.replace('_', ' ').title()}", 0, 'L')
    pdf.set_xy(100, pdf.get_y() + 5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, f"Confidence Score: {confidence_score:.2f}%", 0, 'L')

    # --- TREATMENT SECTION ADDED TO PDF ---
    pdf.set_y(pdf.get_y() + 40) # Position cursor below the image
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommendations', 0, 1, 'L')

    # Overview
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Overview:', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 6, recommendations['overview'], 0, 'L')
    pdf.ln(4)

    # Treatment
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Treatment:', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 6, recommendations['treatment'], 0, 'L')
    pdf.ln(4)

    # Prevention
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Prevention:', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 6, recommendations['prevention'], 0, 'L')

    # --- CLEANUP AND OUTPUT ---
    os.remove(temp_image_path)
    return pdf.output(dest='S').encode('latin-1')


# --- MODEL AND DATA LOADING ---
@st.cache_resource
def load_model_and_indices():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
    model = tf.keras.models.load_model(model_path)
    class_indices_path = os.path.join(working_dir, "class_indices.json")
    with open(class_indices_path) as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_model_and_indices()

# --- IMAGE PROCESSING & PREDICTION FUNCTIONS ---
def load_and_preprocess_image(image_data, target_size=(224, 224)):
    img = Image.open(image_data)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_data, class_indices):
    preprocessed_img = load_and_preprocess_image(image_data)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions) * 100
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence_score


# --- STREAMLIT UI LAYOUT ---
st.markdown('<p class="title">KrishiRakshak 🌿</p>', unsafe_allow_html=True)
st.subheader("Your AI-Powered Plant Health Assistant")
st.markdown("---")

st.header("Get Your Diagnosis in Two Simple Steps")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("1. Upload a Leaf Image")
    uploaded_image = st.file_uploader("Click to upload an image of a plant leaf...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Your Uploaded Leaf Image", use_column_width='auto')

with col2:
    st.subheader("2. Receive Diagnosis & Advice")
    if uploaded_image:
        if st.button('🌿 Classify Leaf Health'):
            with st.spinner('Our AI is analyzing the leaf... Please wait.'):
                prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
                friendly_prediction = prediction.replace('_', ' ').title()

                st.markdown(f'<p class="result-text">Diagnosis: {friendly_prediction}</p>', unsafe_allow_html=True)
                st.info(f"Confidence: {confidence:.2f}%")

                st.markdown("---")
                st.subheader("Treatment Recommendations")

                if "healthy" in prediction:
                    recommendations = treatment_recommendations["default_healthy"]
                else:
                    recommendations = treatment_recommendations.get(prediction, treatment_recommendations["default_disease"])

                st.markdown(f"**Overview:** {recommendations['overview']}")
                st.warning(f"**💊 Treatment:** {recommendations['treatment']}")
                st.success(f"**🛡️ Prevention:** {recommendations['prevention']}")

                pdf_bytes = generate_report(image, prediction, confidence, recommendations)
                st.download_button(
                    label="📥 Download Full Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"KrishiRakshak_Report_{prediction}.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("Upload an image on the left to activate the diagnosis.")

# --- ABOUT SECTION ---
st.markdown("---")
with st.expander("ℹ️ About KrishiRakshak", expanded=False):
    st.write("""
        **KrishiRakshak** is an AI-powered tool designed to help farmers, gardeners, and plant enthusiasts identify plant diseases quickly and accurately from leaf images.
        ### How It Works:
        1.  **Upload an Image:** You provide an image of a plant leaf that appears to be diseased.
        2.  **AI Analysis:** Our application uses a sophisticated deep learning model (CNN) to analyze the visual patterns on the leaf.
        3.  **Get Instant Diagnosis & Advice:** The model predicts the most likely disease and provides tailored treatment and prevention strategies.
        This tool aims to provide a quick, preliminary diagnosis to aid in taking timely and appropriate actions for crop protection. It is not a substitute for professional agricultural advice.
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("Made with ❤️ by an AI enthusiast for a greener world.")
