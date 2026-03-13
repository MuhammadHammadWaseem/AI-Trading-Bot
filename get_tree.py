import os

def generate_tree(dir_path, prefix="", ignore_dirs=('.git', 'venv', '.venv', '__pycache__', 'node_modules', 'saved_models', '.pytest_cache', 'AI Trading Bot.zip')):
    tree_str = ""
    try:
        entries = sorted(os.listdir(dir_path))
    except PermissionError:
        return ""
    
    entries = [e for e in entries if e not in ignore_dirs]
    for i, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        is_last = i == (len(entries) - 1)
        tree_str += prefix + ("└── " if is_last else "├── ") + entry + "\n"
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            tree_str += generate_tree(path, prefix + extension, ignore_dirs)
    return tree_str

if __name__ == "__main__":
    with open("project_tree.txt", "w", encoding="utf-8") as f:
        f.write("AI Trading Bot\n")
        f.write(generate_tree("."))
