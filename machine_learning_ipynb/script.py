import json


def convert_ipynb_to_py():
    ipynb_file = "machine_learning_ipynb\VGG19_final.ipynb"
    py_file = "VGG19.py"
    try:
        # Read the Jupyter Notebook file
        with open(ipynb_file, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        # Extract code cells in order
        code_cells = [
            cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]

        # Write extracted code to Python file
        with open(py_file, "w", encoding="utf-8") as f:
            for cell in code_cells:
                f.write("".join(cell) + "\n\n")

        print(f"Successfully converted '{ipynb_file}' to '{py_file}'")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    convert_ipynb_to_py()
