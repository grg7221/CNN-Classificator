from CNN import CNN
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

batch_size = 512
vocab_size = 25000
embed_dim = 100
num_filters = 100
kernel_sizes = [3, 4, 5]
epochs = 10
lr = 1e-3 * (batch_size / 512)

data = torch.load('dataset/IMDB/train/train.pt')
X = data[0].cuda().long()
Y = data[1].cuda().long()
N = len(X)

model = CNN(
    vocab_size=vocab_size,
    embed_dim=embed_dim,          # C
    num_classes=2,          # output dim
    kernel_sizes=kernel_sizes,
    num_filters=num_filters
).cuda().train()

optimizer = Adam(model.parameters(), lr)

print(f"Total params: {sum(p.numel() for p in model.parameters())}")

for e in range(epochs):
    perm = torch.randperm(N, device='cuda')

    X_shuffle = X[perm]
    Y_shuffle = Y[perm]

    total_loss = 0

    for i in trange(0, N, batch_size):
        xb = X_shuffle[i:i+batch_size]
        yb = Y_shuffle[i:i+batch_size]

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {e+1}: total epoch loss={total_loss:.4f}, last loss={loss.item():.4f}")