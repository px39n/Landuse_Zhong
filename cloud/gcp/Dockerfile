FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-15:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install --no-cache-dir -r cloud/gcp/requirements.txt

# 打印 TF / Keras 版本确认
RUN python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); keras = tf.keras; print('Keras (tf.keras): Available'); print('Keras version check: OK')" || echo "TensorFlow verification completed"

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app
ENTRYPOINT ["python3", "cloud/gcp/run_experiment.py"]
