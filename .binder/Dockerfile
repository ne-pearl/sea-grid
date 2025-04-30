FROM julia:1.11

# Install system dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Install Jupyter and the kernel machinery (bypassing Debian restrictions)
RUN pip3 install --break-system-packages notebook jupyterlab ipykernel

# Set working directory
WORKDIR /home/jovyan

# Copy your repo contents into the container
COPY . /home/jovyan

# Instantiate and precompile your Julia environment
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Register the Julia kernel
RUN julia --project=. -e 'using IJulia; IJulia.installkernel("Julia")'

# Launch JupyterLab instead of classic notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
