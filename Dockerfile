
FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
