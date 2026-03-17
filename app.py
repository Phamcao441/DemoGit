# app.py - OCR PDF scan tiếng Việt ra Word bằng Tesseract + Gemini sửa lỗi

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

# Đường dẫn Tesseract exe (thay nếu cài ở chỗ khác)
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

# Gemini API (google-genai SDK)
GEMINI_API_KEY = "AIzaSyAcSIg-TwvX_SgB0YjYCulB9NkP5wwUlQ8"  # THAY BẰNG KEY THẬT CỦA EM
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "models/gemini-1.5-pro"  # Có thể đổi thành "gemini-1.5-pro" nếu muốn mạnh hơn

def preprocess_image(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    # Adaptive threshold tốt hơn cho scan nhiễu
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(binary)

def pdf_to_text(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=400, poppler_path=POPPLER_PATH)  # Tăng DPI lên 400 để OCR tốt hơn
    except Exception as e:
        raise Exception(f"Lỗi chuyển PDF sang ảnh: {str(e)}")

    full_text = []
    for page_num, img in enumerate(images, 1):
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, lang='vie', config='--psm 6 --oem 3')
        full_text.append(f"Trang {page_num}:\n{text.strip()}\n{'-' * 60}")

    return "\n\n".join(full_text) or "Không nhận dạng được văn bản nào."

def correct_text_with_gemini(raw_text):
    try:
        prompt = f"""
Bạn là chuyên gia sửa lỗi OCR tiếng Việt từ Tesseract. Văn bản gốc thường có lỗi nặng: thiếu dấu thanh (ví dụ 'tuan' thành 'tuần', 'xu ly' thành 'xử lý'), từ ghép bị tách, chữ bị nhận nhầm (ví dụ 'TuzỀ`1n' thành 'Tuần', 'DỄi ưu' thành 'Tối ưu', 'Iịiễm' thành 'Điểm', 'ủn/phấm' thành 'bản/phẩm').

Yêu cầu bắt buộc:
- Sửa lỗi dấu thanh, nối từ ghép, sửa chính tả chuẩn tiếng Việt.
- Giữ nguyên ý nghĩa gốc 100%, không thêm/bớt nội dung, không tự ý diễn giải.
- Giữ nguyên cấu trúc trang: "Trang X:", dấu gạch ngang phân cách.
- Sửa các lỗi phổ biến: 'i' thành 'ì', 'a' thành 'à', 'u' thành 'ủ', từ ghép như 'thông tin' bị tách thành 'thông' 'tin'.
- Nếu phần nào sai quá nặng (ký tự lạ, không đoán được), giữ nguyên phần đó thay vì đoán mò.
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
    doc.add_heading('Kết quả OCR PDF scan (Tesseract + Gemini sửa lỗi)', level=1)
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

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Chỉ hỗ trợ file PDF'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_filename = f"upload_{timestamp}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    file.save(pdf_path)

    try:
        raw_text = pdf_to_text(pdf_path)
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