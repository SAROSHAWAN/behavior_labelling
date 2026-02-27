from spacy import tokens
from collections import deque
from src.config import WINDOW_SIZE, STEP, LAST_INDEX

def clean_persp(target_name, current_doc_id, doc_container, registry):
    cleaned_sentences = []
    doc, context = doc_container[current_doc_id]
    is_last = context.get("is_last", False)
    is_first = (current_doc_id == 0)
    
    # 1. Standard Registry Filter
    doc_refs = [ref for ref in registry[target_name]["references"] if ref["doc_id"] == current_doc_id]

    from collections import defaultdict
    sent_map = defaultdict(list)
    for ref in doc_refs:
        sent_map[ref["local_line"]].append(ref)

    
    
    for sent_idx, sent in enumerate(doc.sents):
        if is_first:
            sent_valid = (0 <= sent_idx < LAST_INDEX)
        elif is_last:
            sent_valid = (sent_idx > 0) # or sent_idx == -1 if using relative indexing
        else:
            sent_valid = (0 < sent_idx < LAST_INDEX)

        # Only perform the surgery if the sentence belongs to this chunk's "unique core"
        if not sent_valid:
            continue

        # 2. FIND CLAUSE HEADS (The new Clause-Level Prefix logic)
        clause_injections = []
        for token in sent:
            # We look for the start of dependent or coordinate clauses
            if token.dep_ in ["advcl", "xcomp", "conj"]:
                # Check 3-breath ancestors to see if it belongs to Target
                if any(a.i in [r["global_start"] for r in doc_refs] for a in list(token.ancestors)[:3]):
                    clause_injections.append({
                        "start": token.idx, # token.idx is spacy's method to get char pos for token
                        "label": f"[{target_name}'s clause] "
                    })

        # 3. get doc-wise position
        sent_text = sent.text
        sent_start_offset = sent.start_char
        
        # Merge your registry refs with your new clause injections
        all_edits = []
        
        # Add registry mentions
        for ref in sent_map[sent_idx]:
            is_poss = ref.get("text", "").lower() in ["his", "her", "its", "their"] or ref.get("type") == "PRP$"
            all_edits.append({
                "start": ref["local_span"][0],
                "label": f"[{target_name}'s] " if is_poss else f"[{target_name}] "
            })
            
        # Add clause prefixes
        all_edits.extend(clause_injections)

        # 4. APPLY EDITS (Reverse order so we dont have to manually keep track of char_pos shifting
        sorted_edits = sorted(all_edits, key=lambda x: x["start"], reverse=True)
        
        for edit in sorted_edits:
            s_local = edit["start"] - sent_start_offset
            # Safety: Ensure s_local is within sentence bounds
            if 0 <= s_local <= len(sent_text):
                sent_text = sent_text[:s_local] + edit["label"] + sent_text[s_local:]

        yield sent_idx, sent_text

"""         cleaned_sentences.append(sent_text)
    return cleaned_sentences """

""" # The Master Store for injected sentences
fixed_sentences = {} 

def populate_fixed_registry(target_name, doc_container, registry):
    for doc_id in range(len(doc_container)):
        # Run your clean_persp logic (the one we just refined)
        sentences = clean_persp(target_name, doc_id, doc_container, registry)
        
        for entry in sentences:
            # Key: (doc_id, local_sent_idx) -> Value: The Injected Text
            fixed_sentences[(entry["doc_id"], entry["sent_local_idx"])] = entry["processed_text"]

def bart_prep_generator(fixed_sentences, doc_container):
    for (doc_id, sent_idx), target_text in fixed_sentences.items():
        doc, context = doc_container[doc_id]
        all_sents = list(doc.sents)
        
        chunk_to_join = []
        
        # 1. Get 2 Sentences Before
        for i in range(sent_idx - 2, sent_idx):
            if i >= 0:
                # Check if we have an injected version of this neighbor
                # If not, use the raw text from the Doc
                chunk_to_join.append(fixed_sentences.get((doc_id, i), all_sents[i].text))
            else:
                # Handle start of book cases: just skip missing neighbors
                pass
        
        # 2. Add the Target Sentence (The Injected one we are currently at)
        chunk_to_join.append(target_text)
        
        # 3. Get 1 Sentence After
        after_idx = sent_idx + 1
        if after_idx < len(all_sents):
            chunk_to_join.append(fixed_sentences.get((doc_id, after_idx), all_sents[after_idx].text))
            
        # 4. Final Join
        final_chunk = " ".join(chunk_to_join).strip()
        
        yield final_chunk """

            
def bart_prep_generator(doc_container, registry, target_name):
    sentence_queue = deque(maxlen=4)
    prompt_prefix = f"Task: Analyze {target_name}. Context: "

    for doc_id, (doc, context) in enumerate(doc_container):
        is_last = context.get("is_last", False)
        is_first = (doc_id == 0)
        all_raw_sents = list(doc.sents)
        
        # 1. INITIALIZE: Engage generator for the first milestone in this doc
        sentence_gen = clean_persp(target_name, doc_id, doc_container, registry)
        
        def get_next_f():
            try:
                # Returns (f_idx, f_text) from your generator
                return next(sentence_gen)
            except StopIteration:
                return float('inf'), None

        f_idx, f_text = get_next_f()

        # 2. INNER LOOP: Chronological s_idx through the doc sentences
        for s_idx, sent in enumerate(doc.sents):
            # Deduplication Check
            if is_first:
                sent_valid = (0 <= s_idx < LAST_INDEX)
            elif is_last:
                sent_valid = (s_idx > 0)
            else:
                sent_valid = (0 < s_idx < LAST_INDEX)

            if not sent_valid:
                continue

            # 3. PUSH LOGIC: Compare s_idx against f_idx
            if s_idx < f_idx:
                # Keep pushing raw s_idx until we hit the milestone
                sentence_queue.append(sent.text)
            
            elif s_idx == f_idx:
                # MILESTONE REACHED: Push f_idx
                sentence_queue.append(f_text)
                
                # RE-ENGAGE: Immediately get new f_idx for the next milestone
                f_idx, f_text = get_next_f()
                
                # 4. THE "LAST CHECK": Push one more to complete the context
                # We peek at s_idx + 1 to provide the 'Post' context
                next_idx = s_idx + 1
                    # If the very next line is also a milestone, push its fixed version
                if next_idx == f_idx:
                    sentence_queue.append(f_text)
                        # Re-engage again because we just consumed this milestone
                    f_idx, f_text = get_next_f()
                else:
                        # Otherwise, push the raw s_idx+1 neighbor
                    sentence_queue.append(all_raw_sents[next_idx].text)

                final_chunk = " ".join(list(sentence_queue)).strip()
                yield f"{prompt_prefix}{final_chunk}"

#TODO: THESE ARE WIP