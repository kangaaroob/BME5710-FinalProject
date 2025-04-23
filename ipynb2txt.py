import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open('base_2.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# Convert to Python script
exporter = PythonExporter()
(source, _) = exporter.from_notebook_node(nb)

# Save as .txt file
with open('codebase.txt', 'w') as f:
    f.write(source)