import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import networkx as nx
import pandas as pd
import json
import os
from itertools import product
from evaluation import train_and_evaluate_model, plot_loss, evaluate_model_on_test, calculate_class_weights
from preprocess import preprocess_data
from graph_build import build_basic_graph, get_nodes_labels_by_id, build_node_to_idx, get_nodes_types
from models import DistanceWeightedGNN, EdgeWeightedGNN, BaselineNodeClassifier, GraphPerBatchDataset
from embedding import generate_embeddings_for_all_graphs
from transformers import BertTokenizer, BertModel

def load_jsonlines(file_path):
    """Load data from a .jsonlist file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

if __name__ == '__main__':
    root_dir = '/home/rize/deep_assignments/NLP'
    # Load the data
    pair_test_df = pd.read_json(f'{root_dir}/CMv data/pair_task/heldout_pair_data.jsonlist', lines=True)
    pair_train_df = pd.read_json(f'{root_dir}/CMv data/pair_task/train_pair_data.jsonlist', lines=True)
    pair_combined_df = pd.concat([pair_test_df, pair_train_df])

    train_period_data = load_jsonlines(f"{root_dir}/CMv data/all/train_period_data.jsonlist")
    heldout_period_data = load_jsonlines(f"{root_dir}/CMv data/all/heldout_period_data.jsonlist")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess the data
    train_processed = preprocess_data(train_period_data)
    test_processed = preprocess_data(heldout_period_data)

    # Extract the positive and negative comment IDs
    positive_comment_ids, negative_comment_ids = get_nodes_types(pair_combined_df)
    all_comments = get_nodes_labels_by_id(pair_combined_df, train_processed, test_processed, positive_comment_ids)

    # Take only the graphs that has at least one delta in the comments
    op_names_set = set(pair_combined_df['op_name'])
    train_processed_reduced = [submission for submission in train_processed if submission['name'] in op_names_set]
    test_processed_reduced = [submission for submission in test_processed if submission['name'] in op_names_set]

    # Build graphs for all conversations in the training data
    conversation_graphs_train = []
    for record in tqdm(train_processed_reduced):
        conversation_graphs_train.append(build_basic_graph(record, record["comments"], with_op=True, all_edges=True))

    # Build graphs for all conversations in the test data
    conversation_graphs_test = []
    for record in tqdm(test_processed_reduced):
        conversation_graphs_test.append(build_basic_graph(record, record["comments"], with_op=True, all_edges=True))

    # # Save the graphs to disk
    # with open(f"{root_dir}/graphs/conversation_graphs_train.json", "w") as f:
    #     json.dump(conversation_graphs_train, f)

    # with open(f"{root_dir}/graphs/conversation_graphs_test.json", "w") as f:
    #     json.dump(conversation_graphs_test, f)

    # Sample
    # sample_train_data = conversation_graphs_train[:20]
    # sample_test_data = conversation_graphs_test[:10]
    node_to_idx = build_node_to_idx(conversation_graphs_train + conversation_graphs_test)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = bert_model.to(device)

    train_node_embeddings, train_op_embeddings = generate_embeddings_for_all_graphs(
        conversation_graphs_train, bert_tokenizer, bert_model, node_to_idx, device
    )

    test_node_embeddings, test_op_embeddings = generate_embeddings_for_all_graphs(
        conversation_graphs_test, bert_tokenizer, bert_model, node_to_idx, device
    )

    # Save the embeddings to disk
    torch.save(train_node_embeddings, f"{root_dir}/embedding/train_node_embeddings.pt")
    torch.save(train_op_embeddings, f"{root_dir}/embedding/train_op_embeddings.pt")
    torch.save(test_node_embeddings, f"{root_dir}/embedding/test_node_embeddings.pt")
    torch.save(test_op_embeddings, f"{root_dir}/embedding/test_op_embeddings.pt")

    with open(f"{root_dir}/embedding/node_to_idx.json", "w") as f:
        json.dump(node_to_idx, f)

    # Split the data into val and test
    train_data, val_data = train_test_split(conversation_graphs_train, test_size=0.1, random_state=42)

    # Load embeddings
    # train_node_embeddings = torch.load(f"{root_dir}/embedding/train_node_embeddings.pt")
    # train_op_embeddings = torch.load(f"{root_dir}/embedding/train_op_embeddings.pt")
    # test_node_embeddings = torch.load(f"{root_dir}/embedding/test_node_embeddings.pt")
    # test_op_embeddings = torch.load(f"{root_dir}/embedding/test_op_embeddings.pt")

    # # Load node-to-index mapping
    # with open(f"{root_dir}/embedding/node_to_idx.json", "r") as f:
    #     node_to_idx = json.load(f)

    labels = {
        comment_id: comment["delta"] if comment["delta"] is not None else 0
        for comment_id, comment in all_comments.items()
    }

    # Create datasets
    train_dataset = GraphPerBatchDataset(
        graphs=train_data,
        node_embeddings=train_node_embeddings,
        op_embeddings=train_op_embeddings,
        labels=labels,
        global_node_to_idx=node_to_idx,
    )

    val_dataset = GraphPerBatchDataset(
        graphs=val_data,
        node_embeddings=test_node_embeddings,
        op_embeddings=test_op_embeddings,
        labels=labels,
        global_node_to_idx=node_to_idx,
    )

    test_dataset = GraphPerBatchDataset(
        graphs=conversation_graphs_test,
        node_embeddings=test_node_embeddings,
        op_embeddings=test_op_embeddings,
        labels=labels,
        global_node_to_idx=node_to_idx,
    )

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=1, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=1),
        "test": DataLoader(test_dataset, batch_size=1),
    }

    # Display a batch from each DataLoader
    for split, loader in dataloaders.items():
        print(f"{split.capitalize()} DataLoader:")
        for batch in loader:
            print(batch)
            break

    train_class_weights = calculate_class_weights(dataloaders["train"], device=device)
    print(f"Class Weights: {train_class_weights}")

    # Define hyperparameters
    learning_rates = [0.001, 0.0001] # 
    num_epochs = [50, 200] # ,  
    weight_decay_values = [1e-5, 1e-3] # 
    dropout_rates = [0.1, 0.3] # 
    model_types = ["distance_weighted_gnn", "edge_weighted_gnn", "baseline"]

    # Variables to track the best model for each type
    best_val_loss_by_model = {model_type: float('inf') for model_type in model_types}
    best_hyperparams_by_model = {model_type: None for model_type in model_types}
    best_model_state_by_model = {model_type: None for model_type in model_types}

    # Iterate over combinations
    for model_type, lr, epochs, weight_decay, dropout in tqdm(product(model_types, learning_rates, num_epochs, weight_decay_values, dropout_rates)):
        print(f"Training with model: {model_type}, lr: {lr}, epochs: {epochs}, l1: {weight_decay}, dropout: {dropout}")

        results_dir = f'{root_dir}/results/{model_type}'
        os.makedirs(results_dir, exist_ok=True)

        # Create a subdirectory for the current hyperparameters
        hyperparam_dir = f"{results_dir}/lr_{lr}_epochs_{epochs}_l1_{weight_decay}_dropout_{dropout}"
        # if os.path.exists(hyperparam_dir):
        #     continue

        os.makedirs(hyperparam_dir, exist_ok=True)

        if model_type == "distance_weighted_gnn":
            model = DistanceWeightedGNN(
                in_channels=train_dataset[0].x.shape[1],
                op_channels=train_dataset[0].op.shape[1],
                out_channels=2,
                dropout=dropout
            )
        elif model_type == "edge_weighted_gnn":
            model = EdgeWeightedGNN( 
                in_channels=train_dataset[0].x.shape[1],
                op_channels=train_dataset[0].op.shape[1],
                out_channels=2,
                dropout=dropout
            )
        elif model_type == "baseline":
            inputs_dim = train_dataset[0].x.shape[1] + train_dataset[0].op.shape[1]
            hidden_dim = 256
            output_dim = 2
            model = BaselineNodeClassifier(inputs_dim, hidden_dim, output_dim, dropout=dropout)

        # Train and evaluate
        train_losses, val_losses, model_best_val = train_and_evaluate_model(
            model, dataloaders, epochs=epochs, lr=lr, class_weights=train_class_weights, weight_decay=weight_decay, device=device
        )

        torch.save(model_best_val, f"{hyperparam_dir}/model_state.pth")

        # Track the best model for the current model type based on validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        if avg_val_loss < best_val_loss_by_model[model_type]:
            best_val_loss_by_model[model_type] = avg_val_loss
            best_model_state_by_model[model_type] = model_best_val
            best_hyperparams_by_model[model_type] = {"lr": lr, "epochs": epochs, "l1_lambda": weight_decay, "dropout": dropout}

        # Plot and save losses for the current model
        plot_loss(train_losses, val_losses, hyperparam_dir)
        evaluate_model_on_test(model, dataloaders["test"], results_dir=hyperparam_dir, device=device)


    # Save the best hyperparameters and models for each model type
    for model_type in model_types:
        print(f"Best Hyperparameters for {model_type}: {best_hyperparams_by_model[model_type]}")
        best_hyperparams_dir = f"{root_dir}/results/{model_type}/best_hyperparams"
        os.makedirs(best_hyperparams_dir, exist_ok=True)

        with open(f"{best_hyperparams_dir}/best_hyperparams.txt", "w") as f:
            f.write(f"Best Hyperparameters: {best_hyperparams_by_model[model_type]}\n")
            f.write(f"Validation Loss: {best_val_loss_by_model[model_type]:.4f}\n")

        torch.save(best_model_state_by_model[model_type], f"{best_hyperparams_dir}/best_model.pth")

        # Re-initialize the model with the best hyperparameters
        if model_type == "distance_weighted_gnn":
            best_model = DistanceWeightedGNN(
                in_channels=train_dataset[0].x.shape[1],
                op_channels=train_dataset[0].op.shape[1],
                out_channels=2,
                dropout=best_hyperparams_by_model[model_type]["dropout"]
            )
        elif model_type == "edge_weighted_gnn":
            best_model = EdgeWeightedGNN(
                in_channels=train_dataset[0].x.shape[1],
                op_channels=train_dataset[0].op.shape[1],
                out_channels=2,
                dropout=best_hyperparams_by_model[model_type]["dropout"]
            )
        elif model_type == "baseline":
            input_dim = train_dataset[0].x.shape[1] + train_dataset[0].op.shape[1]
            hidden_dim = 256
            output_dim = 2
            best_model = BaselineNodeClassifier(
                input_dim, hidden_dim, output_dim,
                dropout=best_hyperparams_by_model[model_type]["dropout"]
            )

        # Load the best model's state
        best_model.load_state_dict(best_model_state_by_model[model_type])
        best_model = best_model.to(device) 

        # Evaluate on the test set 
        evaluate_model_on_test(best_model, dataloaders["test"], results_dir=best_hyperparams_dir, device=device)