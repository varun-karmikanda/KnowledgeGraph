import json
from sklearn.metrics import precision_score, recall_score, f1_score
import spacy

PRONOUNS = {"i", "he", "she", "it", "they", "we", "you", "one", "this", "that", "these", "those", "who", "what", "which", "him", "her", "them", "my", "your", "his", "their", "our", "me"}

# Load predicted and ground truth
with open("outputs/predicted_boundaries.json") as f:
    pred = json.load(f)
with open("data/sentence_boundaries.json") as f:
    gt = json.load(f)

# Make a dict for fast lookup
gt_map = {str(item["sentence_id"]): set(item["nodes"]) for item in gt}
results = []

correct_sent = 0
total_pred = 0
total_gt = 0

nlp = spacy.load("en_core_web_sm")

def extract_entities(sentence):
    doc = nlp(sentence)
    entities = set()
    # Named entities
    for ent in doc.ents:
        entities.add(ent.text.lower())
    # Noun chunks
    for chunk in doc.noun_chunks:
        entities.add(chunk.text.lower())
    return entities

for p in pred:
    sid = str(p["sentence_id"])
    pred_nodes = set(w for w in p["path"] if w.lower() not in PRONOUNS)
    if sid not in gt_map:
        continue
    true_nodes = gt_map[sid]

    # Print mismatches for debugging
    if pred_nodes != true_nodes:
        print(f"Sentence ID: {sid}")
        print(f"Predicted: {pred_nodes}")
        print(f"Ground Truth: {true_nodes}\n")

    # Precision, recall, F1 per sentence
    tp = len(pred_nodes & true_nodes)
    fp = len(pred_nodes - true_nodes)
    fn = len(true_nodes - pred_nodes)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    results.append((precision, recall, f1))

    total_pred += len(pred_nodes)
    total_gt += len(true_nodes)
    if pred_nodes <= true_nodes:
        correct_sent += 1

avg_precision = sum(r[0] for r in results) / len(results)
avg_recall = sum(r[1] for r in results) / len(results)
avg_f1 = sum(r[2] for r in results) / len(results)

print(f"\n🔍 Evaluation Report:")
print(f"📌 Average Precision: {avg_precision:.4f}")
print(f"📌 Average Recall:    {avg_recall:.4f}")
print(f"📌 Average F1 Score:  {avg_f1:.4f}")
print(f"📌 Boundary Precision (fully correct): {correct_sent}/{len(gt)} = {correct_sent/len(gt):.4f}")
print(f"📌 Traversal Efficiency (avg hops): {total_pred/len(pred):.2f}")
print(f"📌 Ground Truth avg length: {total_gt/len(gt):.2f}")
