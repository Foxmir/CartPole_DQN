# Path: project_snapshot_generator.py
# Purpose: Generate a single text snapshot of the repo (structure + file contents) for sharing/debugging.

# Run: PYTHONPATH=. python project_snapshot_generator.py

import os
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).parent
OUTPUT_FILENAME = "project_snapshot.txt"
INCLUDE_EXTENSIONS = ['.py', '.yaml', '.txt', '.md', '.ipynb']

# 1) Directory names to exclude (exact match)
EXCLUDE_DIRS_EXACT = [
    '__pycache__', 
    '.git', 
    '.vscode', 
    'data', 
    'notebooks',
    'wandb',
    '.ipynb_checkpoints',
    'logs',
    'artifacts',
    'plot'
]

# 2) Directory names to exclude by prefix
EXCLUDE_DIRS_PREFIX = [
    'run-', 
    'sweep-', 
    'debug-'
]

# 3) Specific filenames to exclude
EXCLUDE_FILES = [
    OUTPUT_FILENAME,
    'debug.log',
    'secrets.json'
]

# 4) Filenames to exclude by prefix
EXCLUDE_FILES_PREFIX = [
    'test'
]
# --- End configuration ---

def generate_project_snapshot():
    """
    Walk the project directory and write selected files (paths + contents)
    into a single text file.
    """
    output_path = ROOT_DIR / OUTPUT_FILENAME
    script_name = Path(__file__).name
    if script_name not in EXCLUDE_FILES:
        EXCLUDE_FILES.append(script_name)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        print(f"Generating project snapshot to: {output_path}")
        
        for root, dirs, files in os.walk(ROOT_DIR, topdown=True):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS_EXACT]
            dirs[:] = [d for d in dirs if not any(d.startswith(prefix) for prefix in EXCLUDE_DIRS_PREFIX)]
            
            root_path = Path(root)
            
            for file in files:
                # --- Logic ---
                # 1) Build the full file_path first
                file_path = root_path / file
                
                # 2) Then use it for filtering
                # logic: exact filename match OR filename prefix match OR suffix not in allowlist
                if (file in EXCLUDE_FILES or 
                    any(file.startswith(prefix) for prefix in EXCLUDE_FILES_PREFIX) or
                    file_path.suffix not in INCLUDE_EXTENSIONS):
                    continue
                
                # --- Remaining code is unchanged ---
                relative_path = file_path.relative_to(ROOT_DIR)
                
                f_out.write("=" * 80 + "\n")
                f_out.write(f"### File: {relative_path.as_posix()}\n")
                f_out.write("=" * 80 + "\n")
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    f_out.write(content)
                    f_out.write("\n\n")
                    print(f"  [+] Added: {relative_path}")
                except Exception as e:
                    f_out.write(f"*** Failed to read file: {e} ***\n\n")
                    print(f"  [!] Failed to read: {relative_path} ({e})")

    print(
        f"Project snapshot generated: {output_path}. "
        "Note: template/test/secret files are intentionally excluded."
    )

if __name__ == "__main__":
    generate_project_snapshot()
