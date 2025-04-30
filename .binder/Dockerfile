FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    jupyterlab \
    pyomo \
    networkx \
    polars

# Set working directory
WORKDIR /home/jovyan

# Copy everything into the image
COPY . /home/jovyan

# Expose the port for Jupyter (useful locally)
EXPOSE 8888

# Start JupyterLab if running locally
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

#
# To run locally:
# $ docker build -t my-pyomo-env .
# $ docker run -p 8888:8888 -v "$PWD":/home/jovyan my-pyomo-env
#
