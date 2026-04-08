import os
import time
from datetime import datetime
from flask import Flask, request, send_file, render_template, jsonify
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np
from docx import Document
from google import genai
from dotenv import load_dotenv

# Chỉ import các model cần thiết
from models import db, OCRRecord, DublinCoreMetadata

load_dotenv()
client = genai.Client()

app = Flask(__name__)

# --- CẤU HÌNH HỆ THỐNG ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ocr_database.db?timeout=30'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

db.init_app(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

with app.app_context():
    db.create_all()

# Cấu hình đường dẫn Tools
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'D:\poppler\Library\bin'

# ========================= XỬ LÝ OCR =========================

def preprocess_image(img):
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(binary)
    except:
        return img

def get_raw_text(file_path, file_ext):
    """Hàm lấy text thô từ PDF hoặc Ảnh"""
    if file_ext == '.pdf':
        images = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
        full_text = []
        for page_num, img in enumerate(images, 1):
            processed = preprocess_image(img)
            text = pytesseract.image_to_string(processed, lang='vie')
            full_text.append(f"Trang {page_num}:\n{text.strip()}")
        return "\n\n".join(full_text)
    else:
        img = Image.open(file_path)
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, lang='vie')
        return f"Trang 1:\n{text.strip()}"

# ========================= GEMINI SỬA LỖI =========================

def correct_text_with_gemini(raw_text):

    try:

        prompt = f"""

Bạn là chuyên gia sửa lỗi OCR tiếng Việt từ sách scan cũ. Văn bản gốc rất xấu: thiếu dấu thanh, từ tách, ký tự lạ.



Yêu cầu bắt buộc:

- Sửa dấu thanh, nối từ ghép, sửa chính tả chuẩn tiếng Việt.

- Giữ nguyên ý nghĩa gốc 100%, không thêm/bớt nội dung.

- Giữ cấu trúc: "Trang X:", dấu gạch ngang phân cách.

- Sửa lỗi phổ biến: 'tuan' → 'tuần', 'xu ly' → 'xử lý', 'triên khai' → 'triển khai', v.v.

- Không thêm giải thích, markdown, code block. Chỉ trả về văn bản đã sửa.



Văn bản gốc:

{raw_text}



Trả về CHỈ văn bản đã sửa.

"""
        # CÚ PHÁP ĐÚNG - Gemini 3 Flash Preview

        response = client.models.generate_content(

            model="gemini-3-flash-preview",   # ← Tên model đúng

            contents=prompt
        )
        corrected = response.text.strip()
        if corrected.startswith("```") and corrected.endswith("```"):
            corrected = corrected.split("```")[1].strip()
        return corrected

    except Exception as e:

        print(f"Gemini lỗi: {e}")

        return raw_text

# ========================= ROUTES =========================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Nhận dữ liệu Dublin Core từ Form
    dc_title = request.form.get('title', 'Tài liệu không tiêu đề')
    dc_creator = request.form.get('creator', 'Không rõ tác giả')
    dc_subject = request.form.get('subject', 'Chưa phân loại')
    dc_publisher = request.form.get('publisher', 'Hệ thống OCR')
    dc_source = request.form.get('source', 'Bản quét')

    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"up_{timestamp}{file_ext}")
    file.save(file_path)

    try:
        # 1. Chạy OCR lấy text thô (Chỉ dùng để gửi cho Gemini, không lưu DB)
        raw_text = get_raw_text(file_path, file_ext)

        # 2. Gọi Gemini sửa lỗi
        corrected_text = correct_text_with_gemini(raw_text)

        # 3. Lưu Record chính (Chỉ lưu kết quả đã sửa)
        new_record = OCRRecord(
            filename=file.filename,
            input_path=file_path,
            content=corrected_text # Đây là kết quả xịn
        )
        db.session.add(new_record)
        db.session.flush()

        # 4. Lưu Metadata Dublin Core
        dc_metadata = DublinCoreMetadata(
            ocr_id=new_record.id,
            title=dc_title,
            creator=dc_creator,
            subject=dc_subject,
            description=corrected_text[:500] + "...",
            publisher=dc_publisher,
            date=datetime.now().strftime('%Y-%m-%d'),
            type="Text",
            format=file.content_type,
            language="vie",
            source=dc_source
        )
        db.session.add(dc_metadata)
        db.session.commit()

        # 5. Xuất Word
        docx_path = os.path.join(app.config['OUTPUT_FOLDER'], f"Result_{timestamp}.docx")
        doc = Document()
        doc.add_heading(dc_title, 0)
        doc.add_paragraph(corrected_text)
        doc.save(docx_path)

        return send_file(docx_path, as_attachment=True)

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5004)