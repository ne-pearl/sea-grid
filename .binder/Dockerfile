FROM julia:1.11

# Create a Python virtual environment
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv
RUN python3 -m venv /opt/venv

# Activate venv and install Jupyter
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install notebook ipykernel

# Install Julia packages (non-interactive)
COPY Project.toml Manifest.toml ./
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Register Julia kernel inside venv
RUN julia --project=. -e 'using IJulia; IJulia.installkernel("Julia", env=Base.current_project())'

# Set working directory and copy repo contents
WORKDIR /home/jovyan
COPY . /home/jovyan

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
