import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#all_results.get("ZSHOT", {}) -> sbert
#all_results.get("ENCODE", {}) -> bart
def prepare_and_save_chunks(all_res, bart_data_dict, sbert_data_dict, filename="distill_data.pt", chunk_size=100):
    """
    bart_data_dict: { 'Name': [{'label_vector': [...], ...}, ...], ... }
    sbert_data_dict: { 'Name': [ [768_dims], [768_dims], ... ], ... }
    """
    all_encode = []
    all_lbls = []

    for char_name in sbert_data_dict.keys():
        encode_list = sbert_data_dict[char_name]
        lbl_list = [item['weighted_vector'] for item in bart_data_dict[char_name]] #just pick the item we want from bart_dict
        
        N = len(encode_list)
        #right now we skip char if not enough data for the sequence, TODO: we can pad it if we want
        if N < chunk_size:
            continue

        start = 0
        
        while start < N:
            end = start + chunk_size
            
            # --- THE INTEGRATED SHIFT ---
            if end > N:
                # If we would overshoot, shift the window back to hit the exact end
                end = N
                start = N - chunk_size
                
                #final shifted chunk
                all_encode.append(torch.tensor(encode_list[start:end]))
                all_lbls.append(torch.tensor(lbl_list[start:end]))
                break 
            
            # --- THE STANDARD CHUNK ---
            all_encode.append(torch.tensor(encode_list[start:end]))
            all_lbls.append(torch.tensor(lbl_list[start:end]))

            if end == N:
                break
                
            start += chunk_size

    # Convert to high-performance stacked tensors
    sbert_tensor = torch.stack(all_encode).float()
    bart_tensor = torch.stack(all_lbls).float()

    save_path = BASE_DIR / filename
    torch.save({'encodings': sbert_tensor, 'labels': bart_tensor}, save_path)
    
    print(f"Created {sbert_tensor.shape[0]} chunks of 100 windows each.")
    print(f"Saved to: {save_path}")

def load_distill_data(filename="distill_data.pt"):
    load_path = BASE_DIR / filename
    data = torch.load(load_path, weights_only=True)
    return data['encodings'], data['labels']