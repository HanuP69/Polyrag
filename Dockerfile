FROM python:3.9-slim

# Install git and system packages for PyMuPDF/Pytesseract
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Hugging Face runs on port 7860
CMD ["uvicorn", "engine_v4.main:app", "--host", "0.0.0.0", "--port", "7860"]
