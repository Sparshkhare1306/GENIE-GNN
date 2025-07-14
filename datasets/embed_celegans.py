# datasets/embed_celegans.py
import torch
from torch_geometric.nn.models import Node2Vec
from torch_geometric.utils import from_networkx
from tqdm import tqdm

def generate_node2vec_features(graph_nx, embedding_dim=64, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, epochs=50):
    data = from_networkx(graph_nx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node2vec = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        sparse=True
    ).to(device)

    loader = node2vec.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    node2vec.train()
    for epoch in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    node2vec.eval()
    return node2vec.embedding.weight.detach().cpu().numpy()
