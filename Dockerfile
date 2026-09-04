FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubu
RUN pip install --no-cache-dir runpod transformers
COPY handler.py /
CMD ["python", "-u", "/handler.py"]
