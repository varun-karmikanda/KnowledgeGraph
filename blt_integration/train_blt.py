import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from tqdm import tqdm

# Define the BLT model
class BLTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)  # [B, T, D]
        emb = emb.permute(1, 0, 2)  # [T, B, D]
        out = self.transformer(emb)
        out = out.permute(1, 0, 2)  # [B, T, D]
        return self.fc(out)  # [B, T, vocab_size]

# Dataset that skips short sequences
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [seq for seq in sequences if len(seq) > 2]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

# Custom collate function to pad sequences
def pad_collate(batch):
    xs, ys = zip(*batch)
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_pad = pad_sequence(ys, batch_first=True, padding_value=0)
    return xs_pad, ys_pad

if __name__ == "__main__":
    print("[🔁] Training BLT model...")

    # Load sequences
    with open("data/sequences.json") as f:
        sequences = json.load(f)

    # Determine vocab size
    vocab_size = max(max(seq) for seq in sequences) + 1

    # Dataset and Dataloader
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=pad_collate)

    # Model setup
    model = BLTModel(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    for epoch in range(5):  # You can increase epochs
        model.train()
        total_loss = 0
        for x, y in tqdm(dataloader):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "data/blt_model.pt")
    print("[✓] Model saved to data/blt_model.pt")
