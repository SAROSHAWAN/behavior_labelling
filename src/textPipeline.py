from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterator, Tuple, Literal
import difflib
import spacy
spacy.prefer_gpu()
from fastcoref import spacy_component
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'device': 'cuda'})



def make_sentencizer():
    sent_nlp = spacy.blank("en")
    sent_nlp.add_pipe("sentencizer")
    return sent_nlp

sent_nlp = make_sentencizer() #global

def sentenizer(text, sent_nlp): 
    sent_nlp.max_length = max(sent_nlp.max_length, len(text) + 1)
    doc = sent_nlp(text)
    sent_spans = [(s.start_char, s.end_char) for s in doc.sents]
    return sent_spans

#call with a folder in mind, generator for all file in the folder
Mode = Literal["train", "test", "validation"]
def iter_books(mode: Mode, base_dir: str | Path = "data/book", pattern: str = "*.clean.txt"):
    """
    Yields (book_id, text) for each .txt file in:
      data/book/<mode>/
    where mode is one of: train, test, validation
    """
    data_dir = Path(base_dir) / mode

    if not data_dir.exists():
        raise FileNotFoundError(f"Folder not found: {data_dir.resolve()}")

    for fp in sorted(data_dir.glob(pattern)):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        book_id = fp.name.split(".")[0]
        yield book_id, text

def sliding_window(sent_nlp, text, window_size=20, step=17):
    
    sSpans = sentenizer(text, sent_nlp)

    for i in range(0, len(sSpans), step):
        j = min(i + window_size, len(sSpans))
        if i >= j:
            break
        chunk_start = sSpans[i][0] #start of 1st sent
        chunk_end   = sSpans[j-1][1] #end of 20th sent
        chunk_text  = text[chunk_start:chunk_end]

        local_spans = [(s - chunk_start, e - chunk_start) for (s, e) in sSpans[i:j]]

        # Metadata
        context = {
            "offset": chunk_start, 
            "sent_range": {"start_sent": i, "end_sent": j},
            "local_sent_spans": local_spans    
        }

        yield (chunk_text, context)

def get_local_sent_idx(pos: int, local_sent_spans: list[tuple]):
    for idx, (s, e) in enumerate(local_sent_spans):
        if s <= pos < e:
            return idx
    return None  # not found

def score_mention(span):
    """
    Heuristic scoring for a coref mention span.
    - PROPN: +5 (+10 extra if long)
    - PRON: +3
    - else: +1
    - length bonus: +min(10, len(span.text))
    """
    root_pos = span.root.pos_
    n_chars = len(span.text.strip())

    score = 0
    if root_pos == "PROPN":
        score += 5
        if n_chars >= 6:   # "long propn" threshold; tweak as needed
            score += 5
    elif root_pos == "PRON":
        score += 3
    else:
        score += 1

    score += min(5, n_chars)  # small length tie-breaker
    return score

global_ent = []
book_container = [] #propose for book id
cluster_container = []
registry = defaultdict(lambda: {"references": []})


def book_process(text):
    doc_container = []

    #Rename 'offset' to 'context' because it contains the whole dict
    for doc, context in nlp.pipe(sliding_window(sent_nlp, text, window_size=20, step=17), as_tuples=True): #add key to here too in front of in - completed

        # Save BOTH doc and context as a tuple.
        # This prepares the data for the Registry step without needing a re-run, since doc get overwrite with each pipe run
        doc_container.append((doc, context))

        doc_id = len(doc_container) - 1 

        #extract the actual integer offset from the context dictionary
        chunk_start_offset = context["offset"]

        #we start working on the doc imediately to take avantage of its currently being load on living memory so we can take adv of doc_id
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                global_ent.append({
                    "type": "PERSON",
                    "text": ent.text,
                    "global_start": ent.start_char + chunk_start_offset, #key char pos of the 1st char of the ent
                    "global_end": ent.end_char + chunk_start_offset,

                    "doc_id": doc_id,
                    "doc_token_pos": (ent.start, ent.end), #later we can access using span = doc[tok_pos[0]:tok_pos[1]]
                    "sentence_id": get_local_sent_idx(ent.start_char, context["local_sent_spans"])
                })

        #coref
        #redo: now for each cluster, decide the primary person, and store in temp container
        if doc._.coref_clusters:
            for cluster_id, cluster in enumerate(doc._.coref_clusters):
                # SAFETY CHECK 1: Ensure the cluster list itself isn't None
                if cluster is None:
                    continue
                buffer = 0
                # SAFETY CHECK 2: Iterate item by item instead of unpacking immediately
                for item in cluster:
                    if item is None:
                        continue # Skip this specific item if it is None (prevents the crash)

                    start, end = item # Now should be safe to unpack

                    # Map local token indices to global character offsets
                    span = doc.char_span(start, end)
                    score = score_mention(span)
                    if buffer < score:
                        buffer = score
                        primary = (start, end)
            #all we care is: for doc of id x, what clusters it has, and what is the primary of that cluster (tuple position)
            cluster_container.append({
                "doc_id": doc_id,
                "cluster_id": cluster_id,
                "primary": primary
            })

        # Clear RAM
        doc._.trf_data = None

