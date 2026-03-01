import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import matplotlib.pyplot as plt

import streamlit as st

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        rgba(255, 220, 180, 0.25),
        rgba(255, 190, 140, 0.20),
        rgba(255, 160, 100, 0.15)
    );
    background-attachment: fixed;
}

/* SIDEBAR GLASS EFFECT */
section[data-testid="stSidebar"] {
    background: rgba(255, 210, 170, 0.25);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.3);
}

/* Make sidebar content transparent */
section[data-testid="stSidebar"] > div {
    background: transparent;
}

/* Glass Card Effect for Main Blocks */
.block-container {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 2rem;
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-baseweb="input"] > div {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border-radius: 12px;
}

input {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']



@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="jubayer009/retinal_efficientnetv2b3.keras",
        filename="retina_efficientnetv2b3.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()


st.title("Retinal Disease Classification (EfficientNetV2B3)")
with st.sidebar.expander("Model Details"):
    st.markdown("### **EfficientNetV2B3**")
    st.write("Input Shape:", model.input_shape)
    st.write("Output Shape:", model.output_shape)
    st.write("Total Parameters:", f"{model.count_params():,}")
    st.write("Classes:", class_names)
    
st.write("All classes are: ", class_names)

st.markdown("Sample Fundus Retinal Images")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image(Image.open("samples/cataract.jpg"), caption="Cataract", width=150)

with col2:
    st.image(Image.open("samples/diabetic_retinopathy.jpeg"), caption="Diabetic Retinopathy", width=150)

with col3:
    st.image(Image.open("samples/glaucoma.jpg"), caption="Glaucoma", width=150)

with col4:
    st.image(Image.open("samples/normal.jpg"), caption="Normal", width=150)

#st.markdown("**Enter Patient Name:**")
patient_name = st.text_input("", placeholder = "Enter patient name here....")
age = st.text_input("", placeholder = "Enter patient age here....")
gender = st.radio(
    "**Select Gender:**",
    ["**Male**", "**Female**"]
)

uploaded_file = st.file_uploader("Upload Fundus Retinal Image", type=["jpg", "png", "jpeg"])



if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.session_state["uploaded_image"] = image
    left, center, right = st.columns([1,2,1])

    with center:
        st.image(image, caption="Uploaded Image")

    # Create 3 small columns inside center to center button
        b1, b2, b3 = st.columns([1,2,1])
        with b2:
            predict_clicked = st.button("Predict")
        
    
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
   
    if predict_clicked:
    # your prediction code here
        prediction = model.predict(img_array)
        probabilitites = prediction[0]
        class_index = np.argmax(probabilitites)
        predicted_class = class_names[class_index]
        confidence = np.max(probabilitites)
        st.session_state["predicted_class"] = predicted_class
        st.session_state["confidence"] = confidence
        st.session_state["uploaded_image"] = image
        
        st.subheader("Predicted Result")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: {confidence * 100:.2f}%")

        if predicted_class == 'cataract':
            st.markdown("**Suggestion:**")
            st.text("Possible cataract detected. Please consult an ophthalmologist for a detailed eye examination, early treatment or surgery can restore vision effectively.")
        elif predicted_class == "diabetic_retinopathy":
            st.markdown("**Suggestion:**")
            st.text("Indicators of diabetic retinopathy found. Maintain blood sugar control and consult an eye doctor for retinal evaluation and timely treatment.")
        elif predicted_class == "glaucoma":
            st.markdown("**Suggestion:**")
            st.text("Signs of glaucoma detected. Visit an eye specialist as soon as possible for pressure testing and treatment to prevent permanent vision loss.")
        else:
            st.markdown("**Suggestion:**")
            st.text("No major abnormalities detected. Continue regular eye checkups and maintain healthy eye care habits.")

        st.subheader("Disclaimer:")
        st.markdown("**This result is AI-assisted and not a medical diagnosis. Please consult a qualified doctor for confirmation. Remember Ai can make mistakes...! Don't trust it blindly......**")
            
    
        st.subheader("Confidence for All Classes")
    
        fig = plt.figure(figsize=(3.5, 2.2))  # smaller figure

        plt.bar(class_names, probabilitites * 100, width=1)
        
        plt.xticks(rotation=45, fontsize=5)
        plt.yticks(fontsize=5)
        
        plt.xlabel("Classes", fontsize=5)
        plt.ylabel("Confidence (%)", fontsize=5)
        
        plt.ylim(0, 100)
        
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=False)

report_text = f"""
{'Class':<40}{'Precision':<10}{"|  "}{'Recall':<10}{"|  "}{'F1-Score':<10}{"|  "}{'Support':<10}
{'-'*70}
{'cataract':<40}{0.99 :<10.2f}{"|  "}{0.98:<10.2f}{"|  "}{0.98:<10.2f}{"|  "}{100:<10}
{'-'*70}
{'diabetic_retinopathy':<25}{1.00:<10.2f}{"|  "}{1.00:<10.2f}{"|  "}{1.00:<10.2f}{"|  "}{100:<10}
{'-'*70}
{'glaucoma':<40}{0.95:<10.2f}{"|  "}{1.00:<10.2f}{"|  "}{0.98:<10.2f}{"|  "}{100:<10}
{'-'*70}
{'normal':<40}{1.00:<10.2f}{"|  "}{0.96:<10.2f}{"|  "}{0.98:<10.2f}{"|  "}{100:<10}
"""


st.subheader("Model Performance")
st.text("Overall Accuracy: 98.5%")
st.text("Overall Precision: 0.9856")
st.text("Overall Recall: 0.985")
st.text("Overall F1 Score: 0.985")
st.text(report_text)

st.subheader("Model Description")
st.text("We have worked with EfficientNetV2B3 model which is a convolutional neural network architecture that employs fused MBConv blocks and compound scaling to optimize accuracy–efficiency trade-offs while reducing training time. It leverages progressive learning and depth–width–resolution scaling to improve feature representation with fewer parameters. In this work, the model is fine-tuned via transfer learning on retinal fundus images for robust multiclass disease classification.")

#import tempfile
#from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
#from reportlab.lib.styles import getSampleStyleSheet
#from reportlab.lib.units import inch


# ------------------ Suggestion Logic ------------------
def get_suggestion(predicted_class):

    suggestions = {
        "cataract": "Possible cataract detected. Please consult an ophthalmologist for a detailed eye examination, early treatment or surgery can restore vision effectively.",
        
        "diabetic_retinopathy": "Indicators of diabetic retinopathy found. Maintain blood sugar control and consult an eye doctor for retinal evaluation and timely treatment.",
        
        "glaucoma": "Signs of glaucoma detected. Visit an eye specialist as soon as possible for pressure testing and treatment to prevent permanent vision loss.",
        
        "normal": "No major abnormalities detected. Continue regular eye checkups and maintain healthy eye care habits."
    }

    return suggestions.get(predicted_class.lower(), "Consult an eye specialist for further evaluation.")


# ------------------ PDF Generator ------------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime
import tempfile
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
styles = getSampleStyleSheet()
caption_style = ParagraphStyle(
    name="CaptionStyle",
    parent=styles["Normal"],
    fontSize=9,
    textColor=colors.black,
    alignment=1  # center alignment
)

from zoneinfo import ZoneInfo

report_date = datetime.now(ZoneInfo("Asia/Dhaka")).strftime("%d %B %Y, %I:%M %p")

def generate_pdf(predicted_class, confidence, image_path, patient_name, age, report_date, gender):

    suggestion = get_suggestion(predicted_class)

    # Generate current date properly
    #report_date = datetime.now().strftime("%d %B %Y")

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(pdf_path)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("Retinal Disease Classification Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Patient Age:</b> {age}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Patient Gender:</b> {gender}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Date:</b> {report_date}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>Predicted Class:</b> {predicted_class}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>Clinical Suggestion:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(suggestion, styles["Heading3"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(RLImage(image_path, width=3 * inch, height=3 * inch))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(
    f"<i>Figure 1: {predicted_class}</i>",
    caption_style
))
    elements.append(Paragraph("Disclaimer: This result is AI-assisted and not a medical diagnosis. Please consult a qualified doctor for confirmation. Remember Ai can make mistakes... So, Don't trust it blindly....", styles["Heading3"]))

    doc.build(elements)

    return pdf_path
# ------------------ After Prediction ------------------
import tempfile

if "predicted_class" in st.session_state:

    predicted_class = st.session_state["predicted_class"]
    confidence = st.session_state["confidence"]
    image = st.session_state["uploaded_image"]

    if "pdf_file" not in st.session_state:

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file.name)

        pdf_file = generate_pdf(
            predicted_class,
            confidence * 100,
            temp_file.name,
            patient_name,
            age,
            report_date,
            gender
        )

        st.session_state["pdf_file"] = pdf_file

    with open(st.session_state["pdf_file"], "rb") as f:
        st.download_button(
            label="Download Report as PDF",
            data=f,
            file_name=f"{patient_name}_report.pdf",
            mime="application/pdf"
        )


from datetime import datetime
current_year = datetime.now().year

st.markdown(f"""
<div style="text-align:center; padding:15px; background-color:#e9ecef;">
    <strong>Retinal Disease Classification System</strong><br>
    Developed by Jubayer Hossain & Nazia Sultana Marjan<br>
    Department of Computer Science & Engineering<br>
    Daffodil International University<br>
    Contact:
    <a href="mailto:jubayerhossain.cse@gmail.com">
        jubayerhossain.cse@gmail.com
    </a> |
    <a href="mailto:marjan22205101802@diu.edu.bd">
        marjan22205101802@diu.edu.bd
    </a><br>
    © {current_year} Jubayer & Nazia
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br><br><br>", unsafe_allow_html=True)









































































































































