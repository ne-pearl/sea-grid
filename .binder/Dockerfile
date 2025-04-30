FROM julia:1.11-slim

# Install Python + JupyterLab using pip (bypass Debian PEP 668 warning)
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --break-system-packages jupyterlab ipykernel

# Set working directory
WORKDIR /home/jovyan

# Copy your repo contents
COPY . /home/jovyan

# Instantiate your Julia environment
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

# (Optional) Precompile to reduce kernel start time — but increases image size
# RUN julia --project=. -e 'using Pkg; Pkg.precompile()'

# Register the Julia kernel
RUN julia --project=. -e 'using IJulia; IJulia.installkernel("Julia")'

# Clean up to reduce image size
RUN rm -rf ~/.julia/compiled \
    ~/.julia/logs \
    ~/.julia/packages/*/.git \
    /var/lib/apt/lists/* \
    /tmp/* \
    /root/.cache \
    /root/.julia/compiled

# Launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
