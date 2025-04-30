# Use official Julia image
FROM julia:1.11

# Install system dependencies: Python + pip + virtualenv + Jupyter deps
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment (isolates from system Python)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Jupyter inside venv
RUN pip install --no-cache-dir notebook ipykernel

# Install Julia packages and IJulia kernel (lightweight compile mode)
RUN julia --compile=min -e 'using Pkg; Pkg.add("IJulia"); using IJulia; installkernel("Julia")'

# Set working directory to match Binder convention
WORKDIR /home/jovyan
COPY . /home/jovyan

# Use project environment and precompile packages (if any)
RUN julia --compile=min --project=Project.toml -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Jupyter notebook startup
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=", "--NotebookApp.password=", "--allow-root"]
