import json
import torch
import torch.nn as nn
import networkx as nx
import os
from tqdm import tqdm

# --- BLT Model Definition ---
class BLTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)       # [B, T, D]
        out = self.transformer(emb)   # [B, T, D]
        return self.fc(out)           # [B, T, vocab]

    def predict_next(self, seq_tensor):
        with torch.no_grad():
            logits = self.forward(seq_tensor.unsqueeze(0))  # [1, T, vocab]
        return logits.squeeze(0)  # [T, vocab]


# --- Load Model and Data ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("data/sequences.json") as f:
    sequences = json.load(f)
with open("data/svo_triplets.json") as f:
    triplets = json.load(f)
with open("data/node_mapping.json") as f:
    tok_map = json.load(f)

# Ensure token map values are integers
tok_map = {k: int(v) for k, v in tok_map.items()}

# Set vocab size (must match training time)
vocab_size = 256

# Initialize and load model
model = BLTModel(vocab_size).to(device)
model.load_state_dict(torch.load("data/blt_model.pt", map_location=device))
model.eval()

# --- Build Knowledge Graph ---
G = nx.DiGraph()
for t in triplets:
    G.add_edge(t["subject"], t["object"], verb=t["verb"], sent_id=t["sentence_id"])

# --- Entropy-Based Traversal ---
def detect(start_node, G, tok_map, model, threshold=4.0, max_hops=5):
    path = [start_node]
    entropy_vals = []
    hops = 0

    while hops < max_hops:
        token_indices = [tok_map[n] for n in path if n in tok_map and 0 <= tok_map[n] < vocab_size]
        if not token_indices:
            break

        input_seq = torch.tensor(token_indices, dtype=torch.long).to(device)
        logits = model.predict_next(input_seq)
        probs = torch.softmax(logits[-1], dim=-1)
        entropy = -(probs * probs.log()).sum().item()
        entropy_vals.append(entropy)

        if entropy >= threshold:
            break

        successors = list(G.successors(path[-1]))
        valid_succ = [s for s in successors if s in tok_map and 0 <= tok_map[s] < vocab_size]

        if not valid_succ:
            break

        next_node = max(valid_succ, key=lambda x: probs[tok_map[x]].item())
        path.append(next_node)
        hops += 1

    return path, entropy_vals

# --- Sentence Boundary Detection ---
predictions = []
seen_sent_ids = set()

for t in tqdm(triplets, desc="[🔍] Detecting Sentence Boundaries"):
    sid = t["sentence_id"]
    if sid in seen_sent_ids:
        continue
    seen_sent_ids.add(sid)
    start = t["subject"]
    try:
        path, entropy_list = detect(start, G, tok_map, model, threshold=4.0, max_hops=5)
        predictions.append({
            "sentence_id": sid,
            "path": path,
            "entropy": entropy_list
        })
    except Exception as e:
        print(f"[⚠️] Skipped sentence {sid} due to error: {e}")
        continue

# --- Save Predictions ---
os.makedirs("outputs", exist_ok=True)
with open("outputs/predicted_boundaries.json", "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\n[✅] Saved predicted sentence boundaries to outputs/predicted_boundaries.json")
