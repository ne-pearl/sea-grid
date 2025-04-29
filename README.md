# sea-grid

ARROW Summer Education Accelerator

# Getting started

## Configure python

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"  # sanity check
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

## Configure Julia

```julia
import Pkg
Pkg.activate(".")
Pkg.add(["IJulia", "JuMP", "HiGHS", "DataFrames", "CSV"])
Pkg.resolve()
Pkg.instantiate()

using IJulia
notebook()
```
