FROM julia:1.11

# Install Python and pip (needed for Jupyter in Binder)
RUN apt-get update && apt-get install -y python3 python3-pip

# Install Jupyter components
RUN pip3 install notebook ipykernel

# Set the working directory
WORKDIR /home/jovyan

# Copy your repository contents into the image
COPY . /home/jovyan

# Instantiate and precompile your Julia environment
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Register the Julia kernel with Jupyter
RUN julia --project=. -e 'using IJulia; IJulia.installkernel("Julia")'

# Launch Jupyter notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
