import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from annoy_algorithm import AnnoyTree, DataGenerator
import random

# Page configuration
st.set_page_config(
    page_title="ANNOY Algorithm Visualizer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌳 ANNOY Algorithm Visualizer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Approximate Nearest Neighbors Oh Yeah - Integer Data Edition</p>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown('<h3 class="sub-header">🎛️ Controls</h3>', unsafe_allow_html=True)
        
        # Data generation controls
        st.markdown("**📊 Data Generation**")
        data_type = st.selectbox(
            "Data Distribution",
            ["Uniform", "Normal", "Clustered"],
            help="Choose how to generate the random data"
        )
        
        min_val = st.number_input("Minimum Value", value=0, step=1)
        max_val = st.number_input("Maximum Value", value=100, step=1)
        data_count = st.slider("Number of Data Points", 10, 100, 50)
        st.caption("💡 Keep data points under 100 for optimal tree visualization")
        
        # ANNOY parameters
        st.markdown("**🌲 ANNOY Parameters**")
        num_trees = st.slider("Number of Trees", 1, 20, 5)
        search_k = st.slider("Search Results (k)", 1, 20, 5)
        
        # Generate data button
        if st.button("🔄 Generate New Data", use_container_width=True):
            st.session_state.generate_new = True
            
        # Search controls
        st.markdown("**🔍 Search**")
        query_value = st.number_input("Query Value", value=50, step=1)
        
        if st.button("🔎 Search Nearest Neighbors", use_container_width=True):
            st.session_state.perform_search = True
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">📈 Data Visualization</h3>', unsafe_allow_html=True)
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = []
            st.session_state.annoy_tree = None
            st.session_state.generate_new = True
            st.session_state.perform_search = False
            st.session_state.search_results = []
        
        # Generate data if needed
        if st.session_state.generate_new:
            if data_type == "Uniform":
                st.session_state.data = DataGenerator.generate_uniform(min_val, max_val, data_count)
            elif data_type == "Normal":
                mean = (min_val + max_val) // 2
                std = (max_val - min_val) // 6
                st.session_state.data = DataGenerator.generate_normal(mean, std, data_count)
            elif data_type == "Clustered":
                centers = [min_val + (max_val - min_val) * i // 3 for i in range(1, 4)]
                std = (max_val - min_val) // 10
                st.session_state.data = DataGenerator.generate_clusters(centers, std, data_count)
            
            # Build ANNOY trees
            st.session_state.annoy_tree = AnnoyTree(num_trees)
            st.session_state.annoy_tree.build_trees(st.session_state.data)
            st.session_state.generate_new = False
        
        # Display data distribution
        if st.session_state.data:
            df = pd.DataFrame({'values': st.session_state.data})
            
            # Create histogram
            fig_hist = px.histogram(
                df, 
                x='values',
                nbins=min(50, len(set(st.session_state.data))),
                title="Data Distribution",
                labels={'values': 'Integer Values', 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Display statistics
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            with col_stats1:
                st.metric("Total Points", len(st.session_state.data))
            with col_stats2:
                st.metric("Min Value", min(st.session_state.data))
            with col_stats3:
                st.metric("Max Value", max(st.session_state.data))
            with col_stats4:
                st.metric("Mean", f"{np.mean(st.session_state.data):.1f}")
    
    with col2:
        st.markdown('<h3 class="sub-header">🌲 Tree Structure</h3>', unsafe_allow_html=True)
        
        if st.session_state.annoy_tree and st.session_state.annoy_tree.trees:
            # Show tree selector
            tree_index = st.selectbox(
                "Select Tree",
                range(len(st.session_state.annoy_tree.trees)),
                format_func=lambda x: f"Tree {x+1}"
            )
            
            # Get tree structure
            tree_structure = st.session_state.annoy_tree.get_tree_structure(tree_index)
            
            if tree_structure:
                # Display tree info
                st.markdown(f"**Tree {tree_index + 1} Info:**")
                st.markdown(f"- Split value: {tree_structure.get('split_value', 'N/A')}")
                st.markdown(f"- Is leaf: {tree_structure['is_leaf']}")
                
                # Interactive tree visualization
                st.markdown("**Interactive Tree Visualization:**")
                display_interactive_tree(tree_structure)
                
                # Fallback to simple text view
                with st.expander("📝 Text View (Fallback)"):
                    display_tree_simple(tree_structure)
    
    # Search results section
    if st.session_state.perform_search and st.session_state.annoy_tree:
        st.markdown('<h3 class="sub-header">🔍 Search Results</h3>', unsafe_allow_html=True)
        
        # Perform search
        search_results = st.session_state.annoy_tree.search(query_value, search_k)
        st.session_state.search_results = search_results
        
        # Display results
        col_results1, col_results2 = st.columns([1, 2])
        
        with col_results1:
            st.markdown("**Nearest Neighbors:**")
            for i, (value, distance) in enumerate(search_results):
                st.markdown(f"{i+1}. **{value}** (distance: {distance})")
        
        with col_results2:
            # Create scatter plot with search results highlighted
            fig_scatter = go.Figure()
            
            # Plot all data points
            fig_scatter.add_trace(go.Scatter(
                x=list(range(len(st.session_state.data))),
                y=st.session_state.data,
                mode='markers',
                name='All Data',
                marker=dict(color='lightblue', size=8)
            ))
            
            # Highlight search results
            result_values = [result[0] for result in search_results]
            result_indices = [i for i, val in enumerate(st.session_state.data) if val in result_values]
            
            if result_indices:
                fig_scatter.add_trace(go.Scatter(
                    x=result_indices,
                    y=[st.session_state.data[i] for i in result_indices],
                    mode='markers',
                    name='Search Results',
                    marker=dict(color='red', size=12, symbol='star')
                ))
            
            # Highlight query point
            fig_scatter.add_trace(go.Scatter(
                x=[len(st.session_state.data) // 2],
                y=[query_value],
                mode='markers',
                name='Query Point',
                marker=dict(color='green', size=15, symbol='diamond')
            ))
            
            fig_scatter.update_layout(
                title="Data Points with Search Results",
                xaxis_title="Index",
                yaxis_title="Value",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.session_state.perform_search = False

def create_hierarchical_layout(G, pos):
    """Create a proper hierarchical tree layout"""
    # Find the root node (node with no incoming edges)
    root = None
    for node in G.nodes():
        if G.in_degree(node) == 0:
            root = node
            break
    
    if root is None:
        return pos
    
    # Calculate levels and positions
    levels = {}
    visited = set()
    
    def assign_levels(node, level):
        if node in visited:
            return
        visited.add(node)
        
        if level not in levels:
            levels[level] = []
        levels[level].append(node)
        
        # Process children
        children = list(G.successors(node))
        for i, child in enumerate(children):
            assign_levels(child, level + 1)
    
    assign_levels(root, 0)
    
    # Calculate positions
    new_pos = {}
    max_level = max(levels.keys()) if levels else 0
    
    for level, nodes in levels.items():
        # Y position decreases as level increases (root at top)
        y = 1.0 - (level / max_level) if max_level > 0 else 0.5
        
        # X positions are evenly distributed
        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            if num_nodes == 1:
                x = 0.0
            else:
                x = -0.5 + (i / (num_nodes - 1))
            new_pos[node] = (x, y)
    
    return new_pos

def create_tree_visualization(tree_structure, layout_type="hierarchical"):
    """Create an interactive tree visualization with connected nodes"""
    if not tree_structure:
        return None
    
    # Count total nodes to check if tree is too large
    def count_nodes(node):
        count = 1
        if node['children']:
            for child in node['children']:
                count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(tree_structure)
    if total_nodes > 100:
        st.warning(f"Tree has {total_nodes} nodes (max 100). Please reduce data points for better visualization.")
        return None
    
    try:
        import networkx as nx
        
        # Create a NetworkX graph for better layout
        G = nx.DiGraph()
        nodes = []
        edges = []
        current_id = 0
        
        def add_node_to_graph(node, parent_id=None):
            nonlocal current_id
            
            # Create node
            node_id = current_id
            current_id += 1
            
            if node['is_leaf']:
                node_type = "leaf"
                node_label = f"{int(node['value'])}"
                node_color = "#90EE90"  # Light green for leaves
                node_size = 20
            else:
                node_type = "split"
                node_label = f"{int(node['split_value'])}"
                node_color = "#87CEEB"  # Light blue for splits
                node_size = 25
            
            # Add to NetworkX graph
            G.add_node(node_id, 
                      label=node_label,
                      type=node_type,
                      value=node.get('value', node.get('split_value')),
                      color=node_color,
                      size=node_size)
            
            # Add edge from parent if exists
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
            
            # Add children
            if node['children']:
                for child in node['children']:
                    add_node_to_graph(child, node_id)
        
        # Build the graph
        add_node_to_graph(tree_structure)
        
        # Choose layout algorithm
        if layout_type == "hierarchical":
            # Use hierarchical tree layout for clean binary tree structure
            pos = nx.kamada_kawai_layout(G)
            # Convert to hierarchical positioning
            pos = create_hierarchical_layout(G, pos)
        elif layout_type == "spring":
            # Use spring layout for more organic arrangement
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif layout_type == "circular":
            # Use circular layout for compact visualization
            pos = nx.circular_layout(G)
        else:
            # Default to hierarchical layout
            pos = nx.kamada_kawai_layout(G)
            pos = create_hierarchical_layout(G, pos)
        
        # Convert to our format
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            x, y = pos[node_id]
            
            nodes.append({
                'id': node_id,
                'label': node_data['label'],
                'type': node_data['type'],
                'value': node_data['value'],
                'x': x * 1000,  # Scale for better visibility
                'y': y * 1000,
                'color': node_data['color'],
                'size': node_data['size']
            })
        
        # Add edges
        for edge in G.edges():
            edges.append({
                'from': edge[0],
                'to': edge[1],
                'color': '#333333'
            })
        
        return nodes, edges
        
    except ImportError:
        # Fallback to simple layout if networkx is not available
        nodes = []
        edges = []
        current_id = 0
        
        def add_node_to_tree(node, parent_id=None, level=0, x_offset=0):
            nonlocal current_id
            
            # Create node
            node_id = current_id
            current_id += 1
            
            if node['is_leaf']:
                node_type = "leaf"
                node_label = f"{int(node['value'])}"
                node_color = "#90EE90"  # Light green for leaves
                node_size = 20
            else:
                node_type = "split"
                node_label = f"{int(node['split_value'])}"
                node_color = "#87CEEB"  # Light blue for splits
                node_size = 25
            
            nodes.append({
                'id': node_id,
                'label': node_label,
                'type': node_type,
                'value': node.get('value', node.get('split_value')),
                'level': level,
                'x': x_offset,
                'y': level * 150,  # Positive Y so root is at top (will be inverted in display)
                'color': node_color,
                'size': node_size
            })
            
            # Add edge from parent if exists
            if parent_id is not None:
                edges.append({
                    'from': parent_id,
                    'to': node_id,
                    'color': '#333333'
                })
            
            # Add children with proper spacing
            if node['children']:
                num_children = len(node['children'])
                if num_children == 1:
                    # Single child goes directly below
                    child_x = x_offset
                    add_node_to_tree(node['children'][0], node_id, level + 1, child_x)
                else:
                    # Multiple children are spaced evenly
                    total_width = (num_children - 1) * 200  # 200 units between children
                    start_x = x_offset - total_width / 2
                    
                    for i, child in enumerate(node['children']):
                        child_x = start_x + i * 200
                        add_node_to_tree(child, node_id, level + 1, child_x)
            
            return node_id
        
        # Build the tree
        add_node_to_tree(tree_structure)
        
        return nodes, edges

def display_interactive_tree(tree_structure):
    """Display an interactive tree visualization"""
    if not tree_structure:
        st.warning("No tree structure available")
        return
    
    # Layout options
    layout_type = st.selectbox(
        "Tree Layout:",
        ["hierarchical", "spring", "circular"],
        format_func=lambda x: x.title(),
        help="Choose how to arrange the tree nodes"
    )
    
    # Create tree data
    nodes, edges = create_tree_visualization(tree_structure, layout_type)
    
    if not nodes:
        st.warning("Could not create tree visualization")
        return
    
    # Create interactive visualization using Plotly
    import plotly.graph_objects as go
    
    # Create node traces
    node_x = [node['x'] for node in nodes]
    node_y = [node['y'] for node in nodes]
    node_text = [node['label'] for node in nodes]
    node_colors = [node['color'] for node in nodes]
    node_sizes = [node['size'] for node in nodes]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in edges:
        from_node = next(n for n in nodes if n['id'] == edge['from'])
        to_node = next(n for n in nodes if n['id'] == edge['to'])
        
        edge_x.extend([from_node['x'], to_node['x'], None])
        edge_y.extend([from_node['y'], to_node['y'], None])
    
    # Create the figure
    fig = go.Figure()
    
    # Add edges
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color='#333333', width=1.5),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color='#333333', width=2),
            symbol='circle'
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=11, color='black', family='Arial'),
        hoverinfo='text',
        hovertext=node_text,
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title="ANNOY Tree Structure",
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[min(node_x) - 50, max(node_x) + 50]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[min(node_y) - 50, max(node_y) + 50]  # Root at top
        ),
        plot_bgcolor='white',
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    # Display the interactive tree
    st.plotly_chart(fig, use_container_width=True)
    
    # Add tree statistics
    leaf_nodes = [n for n in nodes if n['type'] == 'leaf']
    split_nodes = [n for n in nodes if n['type'] == 'split']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Nodes", len(nodes))
    with col2:
        st.metric("Leaf Nodes", len(leaf_nodes))
    with col3:
        st.metric("Split Nodes", len(split_nodes))
    
    # Add interactive node details
    st.markdown("**Node Details:**")
    selected_node = st.selectbox(
        "Select a node to view details:",
        options=nodes,
        format_func=lambda x: x['label']
    )
    
    if selected_node:
        st.markdown(f"""
        **Node Information:**
        - **Type:** {selected_node['type'].title()}
        - **Value:** {selected_node['value']}
        - **Level:** {selected_node.get('level', 'N/A')}
        - **Position:** ({selected_node['x']:.1f}, {selected_node['y']:.1f})
        """)
        
        # Show children if it's a split node
        if selected_node['type'] == 'split':
            children = [n for n in nodes if any(e['from'] == selected_node['id'] and e['to'] == n['id'] for e in edges)]
            if children:
                st.markdown("**Children:**")
                for child in children:
                    st.markdown(f"- {child['label']}")
        
        # Show parent if it's not the root
        parents = [n for n in nodes if any(e['from'] == n['id'] and e['to'] == selected_node['id'] for e in edges)]
        if parents:
            st.markdown("**Parent:**")
            st.markdown(f"- {parents[0]['label']}")
    
    # Add tree exploration features
    st.markdown("**Tree Exploration:**")
    
    # Filter by node type
    node_type_filter = st.selectbox(
        "Filter by node type:",
        ["All", "Split", "Leaf"]
    )
    
    if node_type_filter != "All":
        filtered_nodes = [n for n in nodes if n['type'].lower() == node_type_filter.lower()]
        st.markdown(f"**{node_type_filter} Nodes:**")
        for node in filtered_nodes:
            st.markdown(f"- {node['label']}")
    
    # Show path to root for selected node
    if selected_node:
        st.markdown("**Path to Root:**")
        path = []
        current_node = selected_node
        
        while current_node:
            path.append(current_node['label'])
            parents = [n for n in nodes if any(e['from'] == n['id'] and e['to'] == current_node['id'] for e in edges)]
            current_node = parents[0] if parents else None
        
        path.reverse()
        st.markdown(" → ".join(path))

def display_tree_simple(node, level=0):
    """Display tree structure in a simple text format (fallback)"""
    indent = "  " * level
    
    if node['is_leaf']:
        st.markdown(f"{indent}**{int(node['value'])}**")
    else:
        st.markdown(f"{indent}**{int(node['split_value'])}**")
        for child in node['children']:
            display_tree_simple(child, level + 1)

if __name__ == "__main__":
    main() 