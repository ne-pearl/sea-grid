FROM julia:1.11

# Install Python and Jupyter (Binder needs this)
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir notebook ipykernel && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /home/jovyan

# Copy only the Julia environment files first to leverage Docker layer caching
COPY Project.toml Manifest.toml ./

# Instantiate only declared dependencies (including IJulia) — no global add
RUN julia --compile=min --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.gc();'

# Then copy the rest of the repo
COPY . .

# Register the IJulia kernel (already installed via project)
RUN julia --compile=min --project=. -e 'using IJulia; installkernel("Julia")'
