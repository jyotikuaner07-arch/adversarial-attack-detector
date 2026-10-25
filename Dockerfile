FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir flask==3.0.0 numpy==1.24.3 matplotlib==3.7.2 Pillow==10.0.0 opencv-python-headless==4.8.0.76 gunicorn
COPY . .
EXPOSE 10000
CMD gunicorn -b 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 300 app:app
