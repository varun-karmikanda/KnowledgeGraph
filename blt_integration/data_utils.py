from sklearn.cluster import KMeans
import numpy as np, pickle, json

def build_tokenizer(embed_path, mapping_path):
    model = pickle.load(open(embed_path, "rb"))  # KarateClub Node2Vec
    mapping = json.load(open(mapping_path))      # node → index

    reverse_mapping = {v: k for k, v in mapping.items()}  # index → node
    vectors = model.get_embedding()  # shape: (num_nodes, embedding_dim)

    km = KMeans(n_clusters=256).fit(vectors)
    return {reverse_mapping[i]: int(km.labels_[i]) for i in range(len(vectors))}

def load_sequences(svo_path, tok_map):
    data = json.load(open(svo_path))
    grouped = {}
    for t in data:
        sid = t["sentence_id"]
        grouped.setdefault(sid, set()).update([t["subject"], t["verb"], t["object"]])
    return [[tok_map[w] for w in sorted(list(seq)) if w in tok_map] for seq in grouped.values()]

if __name__ == "__main__":
    tok_map = build_tokenizer("data/node2vec_model.pkl", "data/node_mapping.json")
    sequences = load_sequences("data/svo_triplets.json", tok_map)

    with open("data/sequences.json", "w") as f:
        json.dump(sequences, f)

    print(f"[✓] Created {len(sequences)} training sequences in data/sequences.json")
