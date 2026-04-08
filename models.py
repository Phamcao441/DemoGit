from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class OCRRecord(db.Model):
    __tablename__ = 'ocr_records'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    input_path = db.Column(db.String(300))
    # Dữ liệu cuối cùng (sau khi Gemini sửa)
    content = db.Column(db.Text, nullable=True) 
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Quan hệ
    dc_metadata = db.relationship('DublinCoreMetadata', backref='record', uselist=False)
    tess_data = db.relationship('TesseractResult', backref='record', uselist=False)

class TesseractResult(db.Model):
    __tablename__ = 'tesseract_results'
    id = db.Column(db.Integer, primary_key=True)
    ocr_id = db.Column(db.Integer, db.ForeignKey('ocr_records.id'), nullable=False)
    raw_content = db.Column(db.Text) # Dữ liệu thô nguyên bản

class DublinCoreMetadata(db.Model):
    __tablename__ = 'dc_metadata'
    id = db.Column(db.Integer, primary_key=True)
    ocr_id = db.Column(db.Integer, db.ForeignKey('ocr_records.id'), nullable=False)
    
    # --- 15 Yếu tố chuẩn Dublin Core (đệ có thể dùng một vài cái chính) ---
    title = db.Column(db.String(300))       # dc.title (Tiêu đề)
    creator = db.Column(db.String(200))     # dc.creator (Tác giả/Người tạo)
    subject = db.Column(db.String(500))     # dc.subject (Chủ đề/Từ khóa)
    description = db.Column(db.Text)        # dc.description (Mô tả/Tóm tắt)
    publisher = db.Column(db.String(200))   # dc.publisher (Nhà xuất bản)
    date = db.Column(db.String(100))        # dc.date (Ngày xuất bản/phát hành)
    type = db.Column(db.String(100))        # dc.type (Loại tài liệu: Text, Image, PDF)
    format = db.Column(db.String(100))      # dc.format (Định dạng file: application/pdf)
    language = db.Column(db.String(50))     # dc.language (Ngôn ngữ: vie, eng)
    source = db.Column(db.String(300))      # dc.source (Nguồn gốc tài liệu)