import json
import sys
import os
from pathlib import Path

# --- DYNAMIC PATH CALCULATION ---
# Get the absolute path of this script (src/processData/lexicon2Json.py)
script_path = Path(__file__).resolve()
# Go up 2 levels: processData -> src -> code (root)
root_dir = script_path.parent.parent.parent
src_dir = script_path.parent.parent

#1. Add 'src' to path so we can import 'config', for testing TODO: if other config call borked, refer to this
sys.path.append(str(src_dir))

# 2. Define Paths using Path objects
empath_data_path = root_dir / ".venv" / "Lib" / "site-packages" / "empath" / "data" / "user"
output_file = root_dir / "data" / "6d_lexicon.json"

categories = ["logic", "perception", "knowledge", "fear", "desire", "stress"]
lexicon_json = {}

for cat in categories:
    file_path = os.path.join(empath_data_path, f"{cat}.empath")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read().split('\t')
            # First element is the name, the rest are words
            lexicon_json[cat] = list(set(content[1:]))

with open(output_file, 'w') as f:
    json.dump(lexicon_json, f, indent=4)

print(f"Success! Your 6D behavior ground truth is now saved to {output_file}")