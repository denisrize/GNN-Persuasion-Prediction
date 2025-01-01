from tqdm import tqdm
import torch

def generate_embeddings_from_graph(graph, tokenizer, model,  device):
    """
    Generate embeddings for each node's text and its corresponding OP text.
    """
    node_embeddings = {}
    op_embeddings = {}
    
    for node, data in graph.nodes(data=True):
        # Text for the node
        node_text = data.get("text", "")
        op_text = data.get("root_body", "")

        # Generate embeddings
        node_embedding = generate_bert_embeddings([node_text], tokenizer, model, device).squeeze(0)
        op_embedding = generate_bert_embeddings([op_text], tokenizer, model, device).squeeze(0)

        node_embeddings[node] = node_embedding
        op_embeddings[node] = op_embedding

    return node_embeddings, op_embeddings

def generate_embeddings_for_all_graphs(graphs, tokenizer, model, node_to_idx, device):
    """
    Generate embeddings for all graphs using the global node-to-index mapping.
    Optimized to generate the OP embedding only once per graph.
    """
    embeddings = torch.zeros(len(node_to_idx), model.config.hidden_size, device=device)
    op_embeddings = torch.zeros(len(node_to_idx), model.config.hidden_size, device=device)

    for graph in tqdm(graphs, desc="Generating embeddings"):
        # Generate the OP embedding once for the entire graph
        root_body = list(graph.nodes(data=True))[0][1].get("root_body", "")  # Assuming root_body is the same for all nodes in the graph
        graph_op_embedding = generate_bert_embeddings([root_body], tokenizer, model, device).squeeze(0)

        for node, data in graph.nodes(data=True):
            node_idx = node_to_idx[node]
            node_text = data.get("text", "")

            # Generate node embedding
            node_embedding = generate_bert_embeddings([node_text], tokenizer, model, device).squeeze(0)

            # Store embeddings
            embeddings[node_idx] = node_embedding
            op_embeddings[node_idx] = graph_op_embedding

    return embeddings, op_embeddings

# Feature Extraction - BERT Embeddings
def generate_bert_embeddings(texts, tokenizer, model, device):
    """Generate embeddings for a list of texts using BERT."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)