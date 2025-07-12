import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLinkPredictorV2(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for layer in self.link_predictor:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        edge_repr = (z[edge_index[0]] + z[edge_index[1]]) / 2
        return self.link_predictor(edge_repr).view(-1)


def load_checkpoint_partial(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model_state']

    # Filter out keys where shapes don't match
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and v.size() == model_dict[k].size()}

    # Update the existing model dict
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(filtered_dict)} layers from checkpoint (partial load)")

# Usage example:
if __name__ == "__main__":
    in_channels = 64  # Set your input feature dimension here
    hidden_channels = 128  # Your model hidden dimension

    model = GCNLinkPredictorV2(in_channels, hidden_channels)
    checkpoint_path = "path/to/your/checkpoint.pt"

    load_checkpoint_partial(model, checkpoint_path)

    # Now you can continue training or evaluation
