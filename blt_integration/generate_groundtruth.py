import json
import os

# Load sequences (tokenized sentences)
with open("data/sequences.json") as f:
    sequences = json.load(f)

# Load token → node name map
with open("data/node_mapping.json") as f:
    tok_map = json.load(f)

# Invert map: token ID (as int) → node name
inv_map = {int(v): k for k, v in tok_map.items()}

# Convert each sentence from token IDs to actual node names
boundaries = []
for i, seq in enumerate(sequences):
    nodes = [inv_map[tok] for tok in seq if tok in inv_map]
    boundaries.append({
        "sentence_id": i,
        "nodes": nodes
    })

# Save to JSON
os.makedirs("data", exist_ok=True)
with open("data/sentence_boundaries.json", "w") as f:
    json.dump(boundaries, f, indent=2)

print("✅ Generated sentence_boundaries.json")
