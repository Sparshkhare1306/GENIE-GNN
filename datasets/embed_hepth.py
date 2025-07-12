from node2vec import Node2Vec
import numpy as np

def generate_node2vec_features(graph, embedding_dim=64, epochs=50):
    # Lowered walk_length and num_walks to reduce memory footprint
    node2vec = Node2Vec(
        graph,
        dimensions=embedding_dim,
        walk_length=20,         # Reduced from default 80
        num_walks=10,           # Reduced from 100
        workers=4               # Parallelize if available
    )

    # Reduced epochs for faster convergence
    model = node2vec.fit(
        window=5,
        min_count=1,
        batch_words=4,
        epochs=epochs
    )

    # Create embedding matrix where row i corresponds to node i
    embeddings = np.zeros((graph.number_of_nodes(), embedding_dim))
    node_id_map = {node: i for i, node in enumerate(graph.nodes())}

    for node in graph.nodes():
        embeddings[node_id_map[node]] = model.wv[str(node)]

    return embeddings
