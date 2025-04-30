FROM julia:1.11

# Install IJulia and any desired packages
RUN julia -e 'using Pkg; Pkg.add(["IJulia"]); Pkg.precompile()'

# Install Jupyter and create a kernel
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install notebook ipykernel
RUN julia -e 'using IJulia; IJulia.installkernel("Julia")'

# Set working directory
WORKDIR /home/jovyan

# Copy everything into the container
COPY . /home/jovyan

# Instantiate the environment
RUN test -f Project.toml && echo "Found Project.toml"
RUN julia --project=Project.toml -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=", "--NotebookApp.password=", "--allow-root"]

