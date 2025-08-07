
import streamlit as st
import fitz  # PyMuPDF
import os
from io import BytesIO
import zipfile

st.set_page_config(page_title="PDF Image Extractor", layout="centered")

st.title("📄 PDF Image Extractor")
st.markdown("Upload a PDF file and extract all high-quality images embedded inside.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    output_folder = "extracted_images"
    os.makedirs(output_folder, exist_ok=True)

    image_paths = []
    image_count = 0

    for page_number in range(len(doc)):
        page = doc[page_number]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page{page_number+1}_img{img_index+1}.{image_ext}"
            filepath = os.path.join(output_folder, image_filename)

            with open(filepath, "wb") as f:
                f.write(image_bytes)
                image_paths.append(filepath)
                image_count += 1

    if image_count == 0:
        st.warning("No images found in the uploaded PDF.")
    else:
        st.success(f"✅ Extracted {image_count} images.")

        # Create a zip of images
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for img_path in image_paths:
                zip_file.write(img_path, arcname=os.path.basename(img_path))
        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download All Images as ZIP",
            data=zip_buffer,
            file_name="extracted_images.zip",
            mime="application/zip"
        )
