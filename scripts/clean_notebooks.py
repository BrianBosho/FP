import os
import nbformat

def clean_notebook(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Clear all outputs
        for cell in nb.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
                
        with open(file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Cleaned {file_path}")
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")

def main():
    # Find all .ipynb files recursively starting from current directory
    # or specific directories if needed.
    # We assume this script is run from the root or federated-gnn root
    
    base_dir = os.getcwd()
    print(f"Searching for notebooks in {base_dir}")
    
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories and wandb
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'wandb']
        
        for file in files:
            if file.endswith('.ipynb'):
                file_path = os.path.join(root, file)
                clean_notebook(file_path)

if __name__ == "__main__":
    main()

