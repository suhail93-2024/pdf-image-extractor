import streamlit as st
import fitz  # PyMuPDF
import os
import io
import zipfile
import tempfile
from PIL import Image
import numpy as np

# OpenCV: use headless build on cloud
import cv2

# Optional: cairosvg for svg -> png conversion
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

st.set_page_config(page_title="PDF Image & Diagram Extractor", layout="centered")

st.title("📄 PDF Image & Diagram Extractor")
st.markdown(
    "Upload a PDF and this tool will extract: 1) embedded images (lossless), "
    "2) page SVGs (vector) and convert to PNG if cairo is available, and "
    "3) raster/line diagrams detected from rendered pages."
)

# ---------- Sidebar options ----------
st.sidebar.header("Extraction options")
zoom = st.sidebar.slider("Render zoom (approx DPI multiplier)", min_value=1.0, max_value=5.0, value=2.0, step=0.5,
                         help="Higher values -> higher DPI renders and larger crops (slower & more memory).")
min_area = st.sidebar.number_input("Min diagram area (px)", min_value=500, max_value=500000, value=2000, step=100,
                                   help="Minimum bounding box area to consider a region a diagram.")
detect_diagrams = st.sidebar.checkbox("Detect & extract diagrams from rendered pages", value=True)
save_svg_png = st.sidebar.checkbox("Convert page SVG -> PNG (cairosvg)", value=CAIROSVG_AVAILABLE)
show_previews = st.sidebar.checkbox("Show small previews in app", value=True)

if save_svg_png and not CAIROSVG_AVAILABLE:
    st.sidebar.warning("cairosvg not available in environment — SVG -> PNG conversion will be skipped.")

# ---------- Helpers ----------
def save_bytes_to_file(b: bytes, path: str):
    with open(path, "wb") as f:
        f.write(b)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def render_page_to_pil(page, zoom_factor=2.0):
    mat = fitz.Matrix(zoom_factor, zoom_factor)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def detect_diagram_boxes(pil_img, min_area_px=2000):
    """Edge-based detection -> contours -> merge -> return list of boxes (x,y,w,h)."""
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dil = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area_px:
            continue
        # skip extremely thin or very small
        if w < 30 or h < 30:
            continue
        # skip box covering nearly whole page (optional)
        if w > 0.98 * w_img and h > 0.98 * h_img:
            continue
        boxes.append((x, y, w, h))
    # merge overlapping / near boxes
    boxes = merge_boxes(boxes, iou_thresh=0.15)
    return boxes

def merge_boxes(boxes, iou_thresh=0.15):
    if not boxes:
        return []
    boxes_arr = np.array(boxes)
    x1 = boxes_arr[:,0]
    y1 = boxes_arr[:,1]
    x2 = x1 + boxes_arr[:,2]
    y2 = y1 + boxes_arr[:,3]
    areas = boxes_arr[:,2] * boxes_arr[:,3]
    idxs = list(range(len(boxes)))
    keep = []
    while idxs:
        i = idxs[0]
        ix1, iy1, ix2, iy2 = x1[i], y1[i], x2[i], y2[i]
        cur_box = [ix1, iy1, ix2, iy2]
        merge_idxs = [i]
        rest = idxs[1:]
        for j in rest:
            jx1, jy1, jx2, jy2 = x1[j], y1[j], x2[j], y2[j]
            inter_x1 = max(ix1, jx1)
            inter_y1 = max(iy1, jy1)
            inter_x2 = min(ix2, jx2)
            inter_y2 = min(iy2, jy2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                inter_area = 0
            else:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union = areas[i] + areas[j] - inter_area
            iou = inter_area / union if union > 0 else 0
            if iou > iou_thresh:
                # expand cur_box
                cur_box[0] = min(cur_box[0], jx1)
                cur_box[1] = min(cur_box[1], jy1)
                cur_box[2] = max(cur_box[2], jx2)
                cur_box[3] = max(cur_box[3], jy2)
                merge_idxs.append(j)
        # remove merged indexes from idxs
        idxs = [k for k in idxs if k not in merge_idxs]
        keep.append((int(cur_box[0]), int(cur_box[1]), int(cur_box[2]-cur_box[0]), int(cur_box[3]-cur_box[1])))
    return keep

def crop_and_save_pil(pil_img, bbox, out_path, padding=8):
    x, y, w, h = bbox
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(pil_img.width, x + w + padding)
    y1 = min(pil_img.height, y + h + padding)
    crop = pil_img.crop((x0, y0, x1, y1))
    crop.save(out_path, format="PNG", optimize=True)

# ---------- File upload UI ----------
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if not uploaded_file:
    st.info("Upload a PDF to begin. Tip: for best results choose documents with clear line diagrams or vector PDFs.")
    st.stop()

# Process file in a temp dir so we don't litter the app workspace
with tempfile.TemporaryDirectory() as tmpdir:
    pdf_path = os.path.join(tmpdir, "input.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    doc = fitz.open(pdf_path)

    output_dir = os.path.join(tmpdir, "extracted")
    ensure_dir(output_dir)

    # Containers for zip packaging and UI listing
    extracted_files = []
    counts = {"embedded": 0, "svg": 0, "svg_png": 0, "diagrams": 0}

    st.info(f"Processing {len(doc)} pages... (render zoom={zoom}, min_area={min_area})")

    # 1) Extract embedded images losslessly
    for p_idx in range(len(doc)):
        page = doc[p_idx]
        images = page.get_images(full=True)
        for img_i, img_meta in enumerate(images):
            xref = img_meta[0]
            b = doc.extract_image(xref)
            img_bytes = b["image"]
            ext = b.get("ext", "png")
            fname = f"page{p_idx+1}_embedded_{img_i+1}.{ext}"
            out_path = os.path.join(output_dir, fname)
            save_bytes_to_file(img_bytes, out_path)
            extracted_files.append(out_path)
            counts["embedded"] += 1

    # 2) Save page SVGs (vector) and optionally convert to PNG
    for p_idx in range(len(doc)):
        page = doc[p_idx]
        try:
            svg = page.get_svg()
            svg_name = f"page{p_idx+1}.svg"
            svg_path = os.path.join(output_dir, svg_name)
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg)
            extracted_files.append(svg_path)
            counts["svg"] += 1

            if save_svg_png and CAIROSVG_AVAILABLE:
                png_name = f"page{p_idx+1}_from_svg.png"
                png_path = os.path.join(output_dir, png_name)
                # Use a reasonable DPI -> multiply default 96 DPI by zoom factor
                cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=png_path, dpi=int(72 * zoom))
                extracted_files.append(png_path)
                counts["svg_png"] += 1
        except Exception as e:
            # If some pages don't produce svg, keep going
            st.warning(f"Warning: failed to get SVG for page {p_idx+1}: {e}")

    # 3) Render pages at high DPI and detect diagrams (if enabled)
    if detect_diagrams:
        for p_idx in range(len(doc)):
            page = doc[p_idx]
            try:
                pil_img = render_page_to_pil(page, zoom_factor=zoom)
            except Exception as e:
                st.warning(f"Failed to render page {p_idx+1}: {e}")
                continue

            boxes = detect_diagram_boxes(pil_img, min_area_px=int(min_area))
            # Fallback: if nothing detected but edges density high, save full page as diagram
            if not boxes:
                gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_ratio = np.sum(edges > 0) / gray.size
                if edge_ratio > 0.006:  # heuristic threshold
                    out_full = os.path.join(output_dir, f"page{p_idx+1}_fullpage_diagram.png")
                    pil_img.save(out_full, format="PNG", optimize=True)
                    extracted_files.append(out_full)
                    counts["diagrams"] += 1
                continue

            for i, bbox in enumerate(boxes):
                out_name = f"page{p_idx+1}_diagram_{i+1}.png"
                out_path = os.path.join(output_dir, out_name)
                crop_and_save_pil(pil_img, bbox, out_path, padding=8)
                extracted_files.append(out_path)
                counts["diagrams"] += 1

    # Build ZIP for download
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in extracted_files:
            # Use basename inside zip
            zf.write(path, arcname=os.path.basename(path))
    zip_buffer.seek(0)

    # ---------- UI output ----------
    st.success("Extraction complete.")
    st.write(f"• Embedded images: **{counts['embedded']}**")
    st.write(f"• Page SVG files: **{counts['svg']}**")
    if save_svg_png:
        st.write(f"• SVG -> PNG converted: **{counts['svg_png']}**")
    st.write(f"• Diagrams detected & saved: **{counts['diagrams']}**")

    if show_previews:
        st.subheader("Previews (small thumbnails)")
        cols = st.columns(3)
        # show up to 12 previews
        previews = extracted_files[:12]
        for i, p in enumerate(previews):
            col = cols[i % 3]
            try:
                # display small thumbnail
                with open(p, "rb") as f:
                    image_bytes = f.read()
                col.image(image_bytes, caption=os.path.basename(p), use_column_width=True)
            except Exception as e:
                col.write(f"Cannot preview {os.path.basename(p)}")

    st.download_button(
        label="📦 Download all extracted files as ZIP",
        data=zip_buffer,
        file_name="extracted_assets.zip",
        mime="application/zip"
    )

    # Also offer individual files list (optional)
    st.markdown("**Files included in the ZIP:**")
    for p in extracted_files:
        st.write("-", os.path.basename(p))

