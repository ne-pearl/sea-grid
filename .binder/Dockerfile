# Dockerfile for Binder + local use (Python version)
FROM python:3.11-slim

# Avoid interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN rm -rf ~/.cache

# Set working directory and copy repo contents
WORKDIR /home/jovyan
COPY . /home/jovyan

# Expose default Jupyter port
EXPOSE 8888

# Launch JupyterLab (Binder will override this CMD)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

#
# To run locally:
# $ docker build -t my-pyomo-env .
# $ docker run -p 8888:8888 -v "$PWD":/home/jovyan my-pyomo-env
#
