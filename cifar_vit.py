
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
import wandb

# todo experiment 1 on wandb : experiment with varying patch size and embed_dim
# todo benchmark m3 mba vs saikat lab pc
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, in_channels):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        # todo compare sine cosine embeds vs learnt embeds
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_len, embed_dim)
        )  # No CLS token

    def forward(self, x):
        return x + self.pos_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=8,
        embed_dim=64,
        in_channels=3,
        num_heads=8,
        num_classes=10,
        depth=6
    ):
        super().__init__()
        self.patch_embeds = PatchEmbedding(embed_dim, patch_size, in_channels)
        self.pos_embeds = PositionEncoding(
            embed_dim, seq_len=(img_size // patch_size) ** 2
        )
        self.transformer_blocks = nn.ModuleList([
            Transformer(embed_dim, num_heads, embed_dim * 4) for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embeds(x)
        x = self.pos_embeds(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.head(x.mean(dim=1))

def train():
    wandb.init()
    cfg = wandb.config

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    ds = CIFAR10(root="./data", train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(ds, 32, True)

    model = VisionTransformer(
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        depth=cfg.depth,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
        epoch_loss = running_loss / len(data_loader)
        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})
        print(f"Epoch [{epoch + 1}/10], Loss: {epoch_loss}")

    wandb.finish()


if __name__ == "__main__":
    train()
