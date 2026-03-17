# ocr_tesseract.py - OCR PDF scan tiếng Việt ra Word bằng Tesseract (cấu trúc sạch, HTML riêng)

import os
from datetime import datetime
from flask import Flask, request, send_file, render_template, jsonify
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np
from docx import Document

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
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

def pdf_to_text(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        raise Exception(f"Lỗi chuyển PDF sang ảnh: {str(e)}")

    full_text = []
    for page_num, img in enumerate(images, 1):
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, lang='vie', config='--psm 6 --oem 3')
        full_text.append(f"Trang {page_num}:\n{text.strip()}\n{'-' * 60}")

    return "\n\n".join(full_text) or "Không nhận dạng được văn bản nào."

def text_to_docx(text_content, output_path):
    doc = Document()
    doc.add_heading('Kết quả OCR PDF scan (Tesseract)', level=1)
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
        extracted_text = pdf_to_text(pdf_path)

        docx_filename = f"ketqua_{timestamp}.docx"
        docx_path = os.path.join(app.config['OUTPUT_FOLDER'], docx_filename)
        text_to_docx(extracted_text, docx_path)

        return send_file(
            docx_path,
            as_attachment=True,
            download_name="KetQua_OCR.docx"
        )

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5004)