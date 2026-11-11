
import torch
import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

from model import SimpleCNN

# Page setup
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    layout="centered"
)

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict its class.")

# Class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load model
@st.cache_resource
def load_model():
    model = SimpleCNN(dropout=0.2)

    checkpoint_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "cifar10_cnn_best.pth"
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )

    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

# Upload image
uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        img_tensor = transform(image).unsqueeze(0)

        with torch.inference_mode():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)

            # Top-3 predictions
            top_probs, top_indices = torch.topk(probs, k=3)

        # Top-1 prediction
        predicted = top_indices[0][0].item()
        confidence = top_probs[0][0].item() * 100

        st.subheader("Prediction Result")

        st.success(f"Class: {class_names[predicted]}")
        st.metric("Confidence", f"{confidence:.2f}%")

        # Top-3 predictions
        st.subheader("Top-3 Predictions")

        for i in range(3):
            class_index = top_indices[0][i].item()
            probability = top_probs[0][i].item() * 100

            st.write(
                f"{class_names[class_index]}: "
                f"{probability:.2f}%"
            )

        # Probability distribution
        st.subheader("Class-wise Probability Distribution")

        probability_df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": probs[0].numpy() * 100
        })

        probability_df = probability_df.sort_values(
            "Probability (%)",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Class")
        )
