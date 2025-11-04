"""
Vision Transformer (ViT) implementation for MNIST classification using PyTorch from
https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
"""
import numpy as np

from tqdm import tqdm, trange
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

logger = logging.getLogger(name=__file__)
logging.basicConfig(level=logging.INFO)

np.random.seed(0)
torch.manual_seed(0)

N_EPOCHS = 5
LR = 0.005

def main():
    """
    prepares MNIST dataset, instantiates a model, and trains for 4 epochs.
    :return:
    """

    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Defining model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"using device: {device}")

    model = MyViT(
        (1, 28, 28),  # MNIST images are 1x28x28
        n_patches=7,
        n_blocks=2,
        hidden_d=8,
        n_heads=2,
        out_d=10  # 10 classes for MNIST
    )
    model.to(device)

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()

    for epoch in trange(N_EPOCHS, desc="Training Epochs"):
        training_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training {epoch+1}", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device) # 32, 1, 28 , 28 and 32
            y_hat = model(x)
            loss = criterion(y_hat, y)
            training_loss += loss.detach().cpu().item()/len(train_loader) # average loss, detach from graph, move to cpu
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"Epoch {epoch+1}/{N_EPOCHS}, Training Loss: {training_loss:.4f}")
        
    # testing loop
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        logger.info(f"Test loss: {test_loss:.2f}")
        logger.info(f"Test accuracy: {correct / total * 100:.2f}%")
        
            

class MyViT(nn.Module):
    def __init__(self, chw=(1,28,28), n_patches=7, hidden_d=8, n_blocks=2, n_heads=2, out_d=10):
        super().__init__()
        self.n_patches = n_patches
        self.chw = chw
        self.hidden_d = hidden_d
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.out_d = out_d # the number of output classes

        c, h, w = self.chw
        assert h == w and h % self.n_patches == 0, "Image must be square and divisible by number of patches"
        self.patch_size = h // self.n_patches
        # linear mapping of patches to hidden dimension
        input_dim = int(c*self.patch_size*self.patch_size)
        self.patch_to_hidden = nn.Linear(input_dim, self.hidden_d)
        # add class token
        self.class_token = nn.Parameter(torch.randn(1, 1, self.hidden_d))
        # add positional embeddings 1, n_patches^2 + 1, hidden_d
        self.positional_embeddings = nn.Parameter(self.get_positional_embeddings(self.n_patches**2 + 1))
        # self.positional_embeddings.requires_grad = False # TODO: double check if ViT learns positional embeddings or not
        # transformer blocks
        self.transformer_blocks = nn.ModuleList([
            MyViTBlock(hidden_d=self.hidden_d, n_heads=self.n_heads) for _ in range(self.n_blocks)
        ])
        # MLP to convert the class token to output classes
        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, self.out_d), nn.Softmax(dim=-1))

    # patching the images
    def patchify(self, images):
        """
        n,c,h,w = n, c, 28, 28, after patching with n_patches=7, we first get (n, n_patches^2, h*w*c/n_patches^2)
        so that for each patch image of size 4x4 (since 28/7=4), we flatten it to a vector of size 16 (4*4)
        in our case, n, 1, 28, 28 -> n, 49, 16
        :param images: a batch of images, shape (n, c, h, w)
        :return:
        """
        n, c, h, w = images.shape
        patches = torch.zeros(n, self.n_patches**2, h*w*c//(self.n_patches**2))
        for idx, image in enumerate(images):
            # image shape: c, h, w
            for i in range(self.n_patches):
                for j in range(self.n_patches):
                    patch = image[:, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size] # c, patch_size, patch_size
                    patches[idx, i*self.n_patches + j, :] = patch.flatten() # flatten to vector c*patch_size*patch_size
        return patches


    def get_positional_embeddings(self, seq_len):
        """
        :param seq_len: length of the sequence, i.e., number of patches + 1 (for class token)
        :return:
        """
        pos_embeddings = torch.zeros(1, seq_len, self.hidden_d)
        for pos in range(seq_len):
            for i in range(self.hidden_d):
                if i % 2 == 0:
                    pos_embeddings[0, pos, i] = np.sin(pos / (10000 ** (i / self.hidden_d)))
                else:
                    pos_embeddings[0, pos, i] = np.cos(pos / (10000 ** ((i - 1) / self.hidden_d)))
        return pos_embeddings


    def forward(self, x):
        """

        :param x: input images, shape (n, c, h, w)
        :return:
        """
        # Forward pass implementation goes here
        patches = self.patchify(x) # n, n_patches^2, patch_dim=c*patch_size*patch_size
        tokens = self.patch_to_hidden(patches) # n, n_patches^2, hidden_d, the path_to_hidden is applied to the last dim, i.e., path_dim -> hidden_d
        # adding class token
        # repeat 5 times of class_token along batch dimension so it becomes n, 1, hidden_d, with tokens.shape=(n, n_patches^2, hidden_d)
        # we do concatenation along dim=1 (sequence dimension) so it becomes n, n_patches^2+1, hidden_d
        tokens = torch.cat([self.class_token.repeat(tokens.shape[0], 1, 1), tokens], dim=1) # n, n_patches^2+1, hidden_d
        # adding positional embeddings, 1, n_patches^2+1, hidden_d, broadcast to n, n_patches^2+1, hidden_d before adding
        output = tokens + self.positional_embeddings # n, n_patches^2+1, hidden_d
        # transformer blocks
        for block in self.transformer_blocks:
            output = block(output) # n, n_patches^2+1, hidden_d
        # MLP head on the class token only
        class_token_final = output[:, 0, :] # n, hidden_d
        output = self.mlp(class_token_final) # n, out_d
        return output

class MyMSA(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super().__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        assert hidden_d % n_heads == 0, "hidden_d must be divisible by n_heads" # this is because we finally need to concatenate all heads back to hidden_d
        self.head_dim = hidden_d // n_heads

        self.q_linear = nn.Linear(hidden_d, hidden_d)
        self.k_linear = nn.Linear(hidden_d, hidden_d)
        self.v_linear = nn.Linear(hidden_d, hidden_d)
        self.out_linear = nn.Linear(hidden_d, hidden_d)

    def forward(self, x):
        n, seq_len, hidden_d = x.shape
        # Linear projections
        Q = self.q_linear(x)  # (n, seq_len, hidden_d)
        K = self.k_linear(x)  # (n, seq_len, hidden_d)
        V = self.v_linear(x)  # (n, seq_len, hidden_d)

        # Reshape for multi-head attention
        Q = Q.view(n, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (n, n_heads, seq_len, head_dim)
        K = K.view(n, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (n, n_heads, seq_len, head_dim)
        V = V.view(n, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (n, n_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (n, n_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)  # (n, n_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, V)  # (n, n_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(n, seq_len, hidden_d)  # (n, seq_len, hidden_d)

        # Final linear layer
        output = self.out_linear(attn_output)  # (n, seq_len, hidden_d)

        return output

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super().__init__()
        self.msa = MyMSA(hidden_d, n_heads)
        self.norm1 = nn.LayerNorm(hidden_d)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_d, hidden_d * 4),
            nn.GELU(),
            nn.Linear(hidden_d * 4, hidden_d)
        )
        self.norm2 = nn.LayerNorm(hidden_d)

    def forward(self, x):
        # Multi-head Self-Attention with residual connection
        attn_output = self.msa(self.norm1(x))
        x = x + attn_output

        # Feed-Forward Network with residual connection
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output

        return x

if __name__ == '__main__':
    # main()
    model = MyViT(
        (1, 28, 28),  # MNIST images are 1x28x28
        n_patches=7,
        n_blocks=2,
        hidden_d=8,
        n_heads=2,
        out_d=10  # 10 classes for MNIST
    )
    x = torch.randn(5, 1, 28, 28) # dummy input
    print(model(x).shape)