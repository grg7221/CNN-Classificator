import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes, num_filters, dropout=0.5):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=num_filters, 
                      kernel_size=k) 
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # B, T, C

        x = x.permute(0, 2, 1) # B, C, T

        conved = [F.relu(conv(x)) for conv in self.convs] # list of (B, n_filters, T-k+1)
        
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved] # list of (B, n_filters)
        
        cat = torch.cat(pooled, dim=1) # B, n_filters * len(kernel_sizes)
        
        cat = self.dropout(cat) # B, n_filters * len(kernel_sizes)
        output = self.fc(cat)   # B, output dim
        
        return output

model = CNN(
    vocab_size=25000,
    embed_dim=100,          # C
    num_classes=2,          # output dim
    kernel_sizes=[3, 4, 5],
    num_filters=100
)

print(sum(p.numel() for p in model.parameters()))