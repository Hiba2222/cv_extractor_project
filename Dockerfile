FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Create necessary directories if they don't exist
RUN mkdir -p /app/data/input \
    /app/data/output \
    /app/data/output1 \
    /app/data/ground_truth \
    /app/data/evaluation \
    /app/data/uploads \
    /app/data/results \
    /app/evaluation_reports

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "WARNING: No requirements.txt found, installing common packages" && \
     pip install --no-cache-dir \
     flask==2.3.3 \
     werkzeug==2.3.7 \
     python-dotenv==1.0.0 \
     PyMuPDF==1.23.7 \
     pdf2image==1.16.3 \
     pillow==10.0.1 \
     google-generativeai==0.3.1 \
     requests==2.31.0 \
     tqdm==4.66.1 \
     pathlib==1.0.1 \
     numpy==1.24.3 \
     pandas==2.0.3 \
     matplotlib==3.7.2 \
     scikit-learn==1.3.0 \
     seaborn==0.12.2)

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=run_web.py
ENV FLASK_ENV=production

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "run_web.py"] 