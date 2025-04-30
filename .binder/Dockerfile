# Base Julia image
FROM julia:1.11.5-bullseye

# Install Python + JupyterLab in a venv (avoids Debian pip restrictions)
RUN apt-get update && apt-get install -y python3 python3-venv python3-pip && \
    python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install JupyterLab and IPython kernel
RUN pip install --no-cache-dir notebook jupyterlab ipykernel

# Install Julia dependencies (assumes Project.toml and Manifest.toml are in the repo)
COPY Project.toml Manifest.toml ./
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Install IJulia and register kernel (uses project environment)
RUN julia --project=. -e 'using IJulia; IJulia.installkernel("Julia")'

# Copy rest of repository (notebooks, code, data, etc.)
COPY . .

# Optional cleanup to reduce image size
RUN rm -rf /root/.julia/logs \
    /root/.julia/packages/*/.git \
    /root/.cache \
    /var/lib/apt/lists/* \
    /tmp/*

# Default command for Binder
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
