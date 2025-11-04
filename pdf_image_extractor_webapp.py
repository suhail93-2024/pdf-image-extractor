# app.py - comprehensive PDF image & diagram extractor
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2, os, io, tempfile, zipfile
import pytesseract
import traceback

# If your tesseract binary is not on PATH, set it here:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="PDF Image & Diagram Extractor", layout="centered")
st.title("PDF Image & Diagram Extractor — embedded, vector & raster figures")
st.markdown("Extract embedded images (lossless), page SVGs, and locate raster/diagram figures using text masking + connected components.")

# Sidebar controls
st.sidebar.header("Settings")
zoom = st.sidebar.slider("Render zoom (1-4) — higher = higher DPI", 1.0, 4.0, 2.0, 0.5)
min_area = st.sidebar.number_input("Min figure area (px)", value=2500, min_value=200, max_value=500000, step=100)
max_candidates = st.sidebar.slider("Max figures per page", 1, 20, 6)
convert_svg = st.sidebar.checkbox("Convert page SVG -> PNG (needs cairosvg)", value=False)
show_preview = st.sidebar.checkbox("Show thumbnails (first 20)", True)

# Helpers
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_bytes(path, b):
    with open(path, "wb") as f:
        f.write(b)

def render_page_to_pil(page, zoom_factor=2.0):
    mat = fitz.Matrix(zoom_factor, zoom_factor)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def extract_embedded_images(doc, out_dir):
    results = []
    for p_idx in range(len(doc)):
        try:
            images = doc[p_idx].get_images(full=True)
        except Exception:
            images = []
        for i_meta, meta in enumerate(images):
            xref = meta[0]
            try:
                b = doc.extract_image(xref)
                ext = b.get("ext", "png")
                name = f"page{p_idx+1}_embedded_{i_meta+1}.{ext}"
                path = os.path.join(out_dir, name)
                save_bytes(path, b["image"])
                results.append(path)
            except Exception:
                continue
    return results

def save_svg_and_optional_png(doc, out_dir, zoom, convert=False):
    results = []
    try:
        import cairosvg
        cairosvg_ok = True
    except Exception:
        cairosvg_ok = False
    for p_idx in range(len(doc)):
        try:
            svg = doc[p_idx].get_svg()
        except Exception:
            svg = None
        if svg:
            svg_name = f"page{p_idx+1}.svg"
            svg_path = os.path.join(out_dir, svg_name)
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg)
            results.append(svg_path)
            if convert and cairosvg_ok:
                try:
                    png_name = f"page{p_idx+1}_from_svg.png"
                    png_path = os.path.join(out_dir, png_name)
                    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=png_path, dpi=int(72*zoom))
                    results.append(png_path)
                except Exception:
                    pass
    return results

# Text mask + connected components approach
def build_text_mask(pil_img, scale_for_ocr=1.0):
    """Use pytesseract to detect text bounding boxes and build a mask (True = text)."""
    rgb = pil_img.convert("RGB")
    if scale_for_ocr != 1.0:
        w,h = rgb.size
        rgb = rgb.resize((int(w*scale_for_ocr), int(h*scale_for_ocr)), Image.LANCZOS)
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Ask tesseract for boxes
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    except Exception:
        # fallback: empty mask (no OCR)
        return np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8), scale_for_ocr
    mask = np.zeros_like(gray, dtype=np.uint8)
    n = len(data.get("left", []))
    for i in range(n):
        try:
            l = int(data["left"][i]); t = int(data["top"][i]); w = int(data["width"][i]); h = int(data["height"][i])
            if w<=0 or h<=0: continue
            cv2.rectangle(mask, (l,t), (l+w, t+h), 255, -1)
        except Exception:
            continue
    return mask, scale_for_ocr

def detect_figures_by_mask_and_cc(pil_img, text_mask, min_area_px=2500, max_candidates=6):
    """From a PIl image and text mask (same size), find non-text connected components likely to be figures."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # morphological clean on text mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # build a candidate mask = areas with strong edges OR color variance, excluding text
    edges = cv2.Canny(gray, 60, 160)
    # color variance: convert to HSV and measure saturation/val
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:,:,1].astype(np.float32)/255.0
    val = hsv[:,:,2].astype(np.float32)/255.0
    color_mask = ((sat > 0.07) | (val < 0.97)).astype(np.uint8)*255
    combined = np.clip(edges.astype(np.uint8) + color_mask, 0, 255)
    # remove text areas
    combined[text_mask>0] = 0
    # morphological close to make solid blobs
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel2, iterations=1)
    # find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area < min_area_px: 
            continue
        # skip very full-page boxes
        if w > 0.95*w_img and h > 0.95*h_img:
            continue
        # also skip slender thin line boxes
        if w < 40 or h < 40:
            continue
        boxes.append((x,y,w,h))
    # merge overlapping boxes
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    # simple IoU-based merge
    merged = []
    for b in boxes:
        bx1,by1,bw,bh = b; bx2 = bx1+bw; by2 = by1+bh
        keep = True
        for i,mb in enumerate(merged):
            mx1,my1,mw,mh = mb; mx2 = mx1+mw; my2 = my1+mh
            inter_x1 = max(bx1, mx1); inter_y1 = max(by1, my1)
            inter_x2 = min(bx2, mx2); inter_y2 = min(by2, my2)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter = (inter_x2-inter_x1)*(inter_y2-inter_y1)
                union = bw*bh + mw*mh - inter
                iou = inter/union if union>0 else 0
                if iou > 0.2:
                    # merge them
                    nx1 = min(bx1, mx1); ny1 = min(by1, my1)
                    nx2 = max(bx2, mx2); ny2 = max(by2, my2)
                    merged[i] = (nx1, ny1, nx2-nx1, ny2-ny1)
                    keep = False
                    break
        if keep:
            merged.append(b)
    # return top-k by area
    merged = sorted(merged, key=lambda b: b[2]*b[3], reverse=True)[:max_candidates]
    return merged

def crop_save(pil_img, bbox, out_path, padding=8):
    x,y,w,h = bbox
    x0 = max(0, x-padding); y0 = max(0, y-padding)
    x1 = min(pil_img.width, x+w+padding); y1 = min(pil_img.height, y+h+padding)
    crop = pil_img.crop((x0,y0,x1,y1))
    crop.save(out_path, format="PNG", optimize=True)

# UI
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if not uploaded:
    st.info("Please upload a PDF file (e.g., the biology PDF).")
    st.stop()

run = st.button("Run extraction")
if not run:
    st.info("Press 'Run extraction' after adjusting settings on the left.")
    st.stop()

# process
with tempfile.TemporaryDirectory() as tmp:
    pdf_path = os.path.join(tmp, "input.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded.read())
    doc = fitz.open(pdf_path)
    out_dir = os.path.join(tmp, "extracted")
    ensure_dir(out_dir)
    extracted = []

    st.write("PyMuPDF version:", getattr(fitz, "__version__", "n/a"))
    st.write(f"Pages: {len(doc)} — rendering zoom: {zoom}x — min_area: {min_area}px")

    # 1) embedded images
    st.info("Extracting embedded images (lossless)...")
    try:
        emb = extract_embedded_images(doc, out_dir)
        extracted += emb
        st.success(f"Embedded images: {len(emb)}")
    except Exception as e:
        st.error("Error extracting embedded images: " + str(e))
        st.write(traceback.format_exc())

    # 2) svg / vector pages
    st.info("Saving page SVGs (if available) and optional conversion...")
    try:
        svg_files = save_svg_and_optional_png(doc, out_dir, zoom, convert=convert_svg)
        extracted += svg_files
        st.success(f"SVGs/pages saved: {len(svg_files)}")
    except Exception as e:
        st.error("SVG save error: " + str(e))

    # 3) Render pages and locate figures using text masking + CC
    st.info("Rendering pages and locating raster diagrams (text masked connected components)...")
    page_count = len(doc)
    for p_idx in range(page_count):
        st.write(f"Processing page {p_idx+1}/{page_count} ...")
        try:
            pil = render_page_to_pil(doc[p_idx], zoom_factor=zoom)
        except Exception as e:
            st.write("Render failed:", e)
            continue

        # 3a) quick auto-extract: try detect images embedded into the page stream as pixmaps rendered in page blocks
        # (we already extracted embedded images object-wise earlier, but sometimes layout contains images not in objects)
        # 3b) build text mask (use smaller scale for OCR speed)
        text_mask, used_scale = build_text_mask(pil, scale_for_ocr=0.6)
        # resize mask to full page size if scaled
        if used_scale != 1.0:
            h_mask, w_mask = text_mask.shape
            text_mask = cv2.resize(text_mask, (pil.width, pil.height), interpolation=cv2.INTER_NEAREST)

        boxes = detect_figures_by_mask_and_cc(pil, text_mask, min_area_px=int(min_area), max_candidates=int(max_candidates))
        if not boxes:
            st.write("No candidate figure boxes found; trying fallback: edge+tile scanning.")
            # fallback: tile-based scanning to detect enclosed boxed illustrations
            gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            # if entire page is edge-dense, treat as fullpage diagram (but prefer cropping using simple layout)
            edge_ratio = np.sum(edges>0) / (gray.size + 1e-9)
            if edge_ratio > 0.018:
                # attempt sub-detection by looking in lower-right and other quadrants
                # check lower-right quadrant for a strong candidate
                hh, ww = gray.shape
                qr = gray[int(hh*0.45):, int(ww*0.45):]
                edges_qr = cv2.Canny(qr,50,150)
                if np.sum(edges_qr>0) / (qr.size + 1e-9) > 0.01:
                    # find bounding boxes in that quadrant
                    cnts, _ = cv2.findContours(edges_qr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    qr_boxes = []
                    for c in cnts:
                        x,y,w,h = cv2.boundingRect(c)
                        area = w*h
                        if area < min_area: continue
                        qr_boxes.append((int(ww*0.45)+x, int(hh*0.45)+y, w, h))
                    if qr_boxes:
                        boxes = sorted(qr_boxes, key=lambda b: b[2]*b[3], reverse=True)[:max_candidates]
            # if still no boxes, possibly no raster figures here
        # Save detected boxes
        for i,b in enumerate(boxes):
            fname = os.path.join(out_dir, f"page{p_idx+1}_figure_{i+1}.png")
            crop_save(pil, b, fname, padding=12)
            extracted.append(fname)

    # Build ZIP
    if not extracted:
        st.warning("No assets found by any method. Try increasing zoom or lowering min_area.")
    else:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in extracted:
                zf.write(p, arcname=os.path.basename(p))
        zip_buf.seek(0)
        st.success(f"Extraction done — {len(extracted)} items")
        if show_preview:
            st.subheader("Preview (first 20)")
            cols = st.columns(3)
            for i, fpath in enumerate(extracted[:20]):
                try:
                    cols[i%3].image(fpath, caption=os.path.basename(fpath), use_container_width=True)
                except Exception as e:
                    cols[i%3].write(os.path.basename(fpath))
        st.download_button("Download all extracted files", data=zip_buf, file_name="extracted_assets.zip", mime="application/zip")
        st.write("Files:")
        for p in extracted:
            st.write("-", os.path.basename(p))
