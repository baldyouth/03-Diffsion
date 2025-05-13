import os

def print_directory_tree(root_dir, indent=""):
    print(f"{indent}{os.path.basename(root_dir)}/")
    indent += "│   "
    for i, item in enumerate(sorted(os.listdir(root_dir))):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            print_directory_tree(path, indent)
        else:
            print(f"{indent}├── {item}")