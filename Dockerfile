FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py advice.py gradcam.py class_names.json plant_disease_mobilenet.keras ./

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
