from collections import defaultdict
import networkx as nx

def get_nodes_types(pair_data):
    positive_comment_ids = set()
    negative_comment_ids = set()
    for _, row in pair_data[['positive', 'negative']].iterrows():  # Iterate through rows
        positives = row['positive']  # Access 'positive' column
        negatives = row['negative']  # Access 'negative' column
        for comment in positives["comments"]:
            positive_comment_ids.add(comment["id"])
        for comment in negatives["comments"]:
            negative_comment_ids.add(comment["id"])
    
    return positive_comment_ids, negative_comment_ids
    
def get_nodes_labels_by_id(pair_combined_df, train_processed, test_processed, positive_comment_ids):
    all_comments = {}
    op_names_set = set(pair_combined_df['op_name'])

    for i, submission in enumerate(train_processed):
        if submission['name'] in op_names_set:
            all_comments[submission['id']] = submission
            all_comments[submission['id']]["delta"] = 0 

            for comment in submission["comments"]:
                comment_id = comment["id"]
                if comment_id not in all_comments:  # Avoid overwrites
                    all_comments[comment_id] = comment
                # Assign delta label
                all_comments[comment_id]["delta"] = 1 if comment_id in positive_comment_ids else 0

    for submission in test_processed:
        if submission['name'] in op_names_set:
            all_comments[submission['id']] = submission
            all_comments[submission['id']]["delta"] = 0 
            for comment in submission["comments"]:
                comment_id = comment["id"]
                if comment_id not in all_comments:  # Avoid overwrites
                    all_comments[comment_id] = comment
                # Assign delta label
                all_comments[comment_id]["delta"] = 1 if comment_id in positive_comment_ids else 0

    return all_comments

# Graph Construction
def standardize_node_id(node_id):
    """Remove Reddit-specific prefixes (t1_, t3_) from node IDs."""
    return node_id.split("_")[-1]

def build_basic_graph(root, comments, comment_to_include=None, all_edges=False, filter_distance=1):
    """Build a conversation tree graph where each parent points to all its descendants."""
    graph = nx.DiGraph()
    
    # Filter comments and add the root node
    comment_ids, root_id, root_author, root_body = initialize_graph(graph, root, comments)

    # Add comment nodes and direct parent-child edges
    delta_nodes = add_nodes_and_edges(graph, comments, root_id, root_author, root_body, comment_ids, comment_to_include)

    # Calculate distances from root to delta nodes
    delta_distances = calculate_delta_distances(graph, root_id, delta_nodes)

    # Add all descendant edges if specified
    if all_edges:
        add_descendant_edges(graph, root_id)
        add_longest_path_from_root(graph, root_id, filter_distance)
        
    # Update root node attributes
    update_root_node_attributes(graph, root_id, delta_nodes, delta_distances)

    return graph


def initialize_graph(graph, root, comments):
    """Initialize the graph by adding the root node and filtering comments."""
    comment_ids = {
        standardize_node_id(comment["id"]) for comment in comments if "author" not in comment or comment["author"] != 'DeltaBot'
    }
    root_id = standardize_node_id(root["id"])
    root_author = root.get("author", "Unknown")
    root_body = root.get("body", "")
    
    graph.add_node(
        root_id,
        text=root.get("body", ""),
        delta=0,
        root_id=root_id,
        root_body=root_body,
        author=root_author,
        ups=root.get("ups", 0),
        downs=root.get("downs", 0),
        mask=1,
        author_flair_text=root.get("author_flair_text", 0),
        distance_to_op=0,
        delta_count=0,  # Initialize delta count
        delta_distances=[],  # Initialize list for distances to delta nodes
    )
    comment_ids.add(root_id)

    return comment_ids, root_id, root_author, root_body

def add_nodes_and_edges(graph, comments, root_id, root_author, root_body, comment_ids, comment_to_include=None):
    """Add nodes and direct parent-child edges to the graph."""
    delta_nodes = []
    
    for comment in comments:
        # Standardize node IDs
        node_id = standardize_node_id(comment["id"])
        parent_id = standardize_node_id(comment.get("parent_id", ""))
        
        # Skip DeltaBot comments
        if node_id not in comment_ids:
            continue

        # Add the current comment as a node
        comment_author = comment.get("author", "Unknown")
        delta_value = comment.get("delta", 0)

        # Check if the comment should be masked
        if (comment_to_include and node_id not in comment_to_include) or comment_author == root_author:
            mask_node = 1
        else:
            mask_node = 0

        graph.add_node(
            node_id,
            text=comment.get("body", ""),
            delta=delta_value,
            root_id=root_id,
            root_body=root_body,
            author=comment_author,
            ups=comment.get("ups", 0),
            downs=comment.get("downs", 0),
            mask=mask_node, # mask nodes that are not in the comment_to_include list
            author_flair_text=comment.get("author_flair_text", 0),
        )

        # Track delta nodes
        if delta_value == 1:
            delta_nodes.append(node_id)

        # Add direct parent-child edge with distance = 1
        if parent_id and parent_id in comment_ids:
            graph.add_edge(parent_id, node_id, distance=1)
    
    return delta_nodes

def calculate_delta_distances(graph, root_id, delta_nodes):
    """Calculate the distances from the root node to delta nodes."""
    delta_distances = []
    for delta_node in delta_nodes:
        try:
            distance = nx.shortest_path_length(graph, source=root_id, target=delta_node)
            delta_distances.append(distance)
        except nx.NetworkXNoPath:
            pass  # Skip if no path exists
    return delta_distances

def add_descendant_edges(graph, root_id):
    """Add edges from each node to all its descendants."""
    for node in list(graph.nodes):
        descendants = nx.descendants(graph, node)  # Get all descendants
        for descendant in descendants:
            if not graph.has_edge(node, descendant):  # Avoid duplicate edges
                distance = nx.shortest_path_length(graph, source=node, target=descendant)
                graph.add_edge(node, descendant, distance=distance)  # Add distance as an attribute

def add_longest_path_from_root(graph, root_id, filter_distance=1):
    """
    Compute the longest path from the root node to every other node in the graph.
    Assumes that each node has direct edges to its descendants.
    """
    # Initialize the longest path dictionary
    longest_path_distances = {node: float("-inf") for node in graph.nodes}
    longest_path_distances[root_id] = 0  # Distance to root is 0

    # Perform topological sort for proper traversal order
    try:
        topo_sort = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("Graph must be a DAG to compute longest paths.")

    # Traverse in topological order to compute the longest path
    for node in topo_sort:
        for _, descendant, edge_data in graph.out_edges(node, data=True):
            distance = edge_data.get("distance", 1)  # Default to 1 if no distance attribute
            # Update the longest path distance if a longer path is found
            longest_path_distances[descendant] = max(
                longest_path_distances[descendant],
                longest_path_distances[node] + distance
            )

    # Add the longest path distance as a node attribute
    for node, longest_distance in longest_path_distances.items():
        graph.nodes[node]["distance_to_op"] = longest_distance
        # Mask nodes that are beyond the filter distance
        if longest_distance < filter_distance:
            graph.nodes[node]["mask"] = 1

    return graph

def update_root_node_attributes(graph, root_id, delta_nodes, delta_distances):
    """Update the root node with delta count and distances."""
    graph.nodes[root_id]["delta_count"] = len(delta_nodes)
    graph.nodes[root_id]["delta_distances"] = delta_distances

def build_node_to_idx(graphs):
    """
    Create a global mapping from node IDs to numerical indices.
    """
    node_to_idx = {}
    idx = 0
    for graph in graphs:
        for node in graph.nodes:
            if node not in node_to_idx:
                node_to_idx[node] = idx
                idx += 1
    return node_to_idx