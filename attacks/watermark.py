import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

def apply_watermark(graph, features, model, device):
    print("Starting training with watermark (remapping nodes)...")

    # Create a mapping from node IDs to contiguous indices [0, 1, ..., num_nodes - 1]
    node_id_map = {node_id: i for i, node_id in enumerate(graph.nodes())}

    # Remap edges to use contiguous node indices
    remapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()

    # Remap features to be in the same order as new indices
    sorted_node_ids = sorted(graph.nodes())
    reverse_map = {node_id: i for i, node_id in enumerate(sorted_node_ids)}
    perm = [node_id_map[nid] for nid in sorted_node_ids]
    x = features[perm]

    # Create Data object
    data = Data(x=x, edge_index=edge_index)

    # Split data
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)
    print(f"train_data has attributes: {train_data.keys()}")

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(50):
        optimizer.zero_grad()
        z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))
        loss = model.recon_loss(z, train_data.edge_label_index.to(device), train_data.edge_label.to(device))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model
