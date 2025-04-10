import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open('UNet.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# Convert to Python script
exporter = PythonExporter()
(source, _) = exporter.from_notebook_node(nb)

# Save as .txt file
with open('unet.txt', 'w') as f:
    f.write(source)
