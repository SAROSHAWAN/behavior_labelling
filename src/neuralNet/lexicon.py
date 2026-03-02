import json
from pathlib import Path
from empath import Empath

lexicon = Empath()

CATEGORIES = {
    "logic": ["because", "therefore", "reason", "conclusion", "logic", "deduce"],
    "perception": ["observe", "notice", "see", "hear", "detect", "scent"],
    "knowledge": ["know", "fact", "evidence", "aware", "certainty", "truth"],
    "fear": ["afraid", "horror", "dread", "terror", "fright", "scared"],
    "desire": ["want", "hope", "ambition", "wish", "longing", "yearn"],
    "stress": ["pressure", "tense", "burden", "strained", "grief", "anxious"]
}


for category_name, seed_words in CATEGORIES.items():
    result = lexicon.create_category(category_name, seed_words, model="fiction", size=200)


