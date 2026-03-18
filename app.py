# app.py - OCR PDF/Ảnh scan tiếng Việt ra Word bằng Tesseract + Gemini sửa lỗi

import os
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

# Load biến từ file .env (ẩn key an toàn)
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Không tìm thấy GEMINI_API_KEY trong file .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "models/gemini-1.5-flash"  # Đổi thành "models/gemini-1.5-pro" nếu muốn chính xác hơn

# Đường dẫn Tesseract exe (thay nếu khác)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Đường dẫn Poppler (thay nếu khác)
POPPLER_PATH = r'D:\poppler\Library\bin'

def preprocess_image(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    # Denoise nhẹ
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(binary)

def pdf_to_text(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=400, poppler_path=POPPLER_PATH)
    except Exception as e:
        raise Exception(f"Lỗi chuyển PDF sang ảnh: {str(e)}")

    full_text = []
    for page_num, img in enumerate(images, 1):
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, lang='vie', config='--psm 6 --oem 3')
        full_text.append(f"Trang {page_num}:\n{text.strip()}\n{'-' * 60}")

    return "\n\n".join(full_text) or "Không nhận dạng được văn bản nào."

def image_to_text(image_path):
    try:
        img = Image.open(image_path)
        # Resize 1.5x để OCR tốt hơn mà không chậm
        img = img.resize((int(img.width * 1.5), int(img.height * 1.5)), Image.LANCZOS)
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, lang='vie', config='--psm 6 --oem 3')
        return f"Trang 1 (ảnh đơn):\n{text.strip()}\n{'-' * 60}"
    except Exception as e:
        raise Exception(f"Lỗi OCR ảnh: {str(e)}")

def correct_text_with_gemini(raw_text):
    try:
        prompt = f"""
Bạn là chuyên gia sửa lỗi OCR tiếng Việt từ sách scan cũ. Văn bản gốc rất xấu: thiếu dấu thanh, từ tách, ký tự lạ (ví dụ 'TuzỀ`1n' → 'Tuần', 'DỄi ưu' → 'Tối ưu', 'Iịiễm' → 'Điểm', 'ủn/phấm' → 'bản/phẩm', 'xứ lý' → 'xử lý', 'triên khai' → 'triển khai').

Yêu cầu bắt buộc:
- Sửa dấu thanh, nối từ ghép, sửa chính tả chuẩn tiếng Việt.
- Giữ nguyên ý nghĩa gốc 100%, không thêm/bớt nội dung, không tự ý diễn giải.
- Giữ cấu trúc: "Trang X:", dấu gạch ngang phân cách.
- Sửa lỗi phổ biến sách scan: 'tuan' → 'tuần', 'xu ly' → 'xử lý', 'lôi' → 'lỗi', 'khăc phục' → 'khắc phục'.
- Nếu chữ bị nhận nhầm nặng (ký tự lạ), giữ nguyên phần đó.
- Không thêm giải thích, markdown, code block, chỉ trả về văn bản đã sửa.

Văn bản gốc:
{raw_text}

Trả về CHỈ văn bản đã sửa.
"""

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt
        )

        corrected = response.text.strip()

        if corrected.startswith("```") and corrected.endswith("```"):
            corrected = corrected.split("```")[1].strip()

        return corrected
    except Exception as e:
        print(f"Gemini lỗi: {e}")
        return raw_text

def text_to_docx(text_content, output_path):
    doc = Document()
    doc.add_heading('Kết quả OCR (Tesseract + Gemini sửa lỗi)', level=1)
    for part in text_content.split('\n'):
        if not part.strip():
            continue
        if part.startswith('Trang'):
            doc.add_heading(part, level=2)
        elif part.startswith('-'):
            doc.add_paragraph(part)
        else:
            doc.add_paragraph(part)
    doc.save(output_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed_ext = {'.pdf', '.png', '.jpg', '.jpeg'}
    if file_ext not in allowed_ext:
        return jsonify({'error': 'Chỉ hỗ trợ file PDF, PNG, JPG, JPEG'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"upload_{timestamp}{file_ext}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        if file_ext == '.pdf':
            raw_text = pdf_to_text(file_path)
        else:
            raw_text = image_to_text(file_path)

        corrected_text = correct_text_with_gemini(raw_text)

        docx_filename = f"ketqua_{timestamp}.docx"
        docx_path = os.path.join(app.config['OUTPUT_FOLDER'], docx_filename)
        text_to_docx(corrected_text, docx_path)

        return send_file(
            docx_path,
            as_attachment=True,
            download_name="KetQua_OCR_Gemini.docx"
        )

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5004)