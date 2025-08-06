#!/usr/bin/env python3
"""
Tree Visualization Demo
This script demonstrates the enhanced tree visualization features with different layouts.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from annoy_algorithm import AnnoyTree, DataGenerator

# Page configuration
st.set_page_config(
    page_title="Tree Visualization Demo",
    page_icon="🌳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .demo-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
            pos = nx.kamada_kawai_layout(G)
            # Convert to hierarchical positioning
            pos = create_hierarchical_layout(G, pos)
        elif layout_type == "spring":
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif layout_type == "circular":
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
                'x': x * 1000,
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
        # Fallback layout
        nodes = []
        edges = []
        current_id = 0
        
        def add_node_to_tree(node, parent_id=None, level=0, x_offset=0):
            nonlocal current_id
            
            node_id = current_id
            current_id += 1
            
            if node['is_leaf']:
                node_type = "leaf"
                node_label = f"{int(node['value'])}"
                node_color = "#90EE90"
                node_size = 20
            else:
                node_type = "split"
                node_label = f"{int(node['split_value'])}"
                node_color = "#87CEEB"
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
        
        add_node_to_tree(tree_structure)
        return nodes, edges

def display_tree_visualization(nodes, edges, layout_type):
    """Display the tree visualization"""
    if not nodes:
        st.warning("No nodes to display")
        return
    
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
        title=f"ANNOY Tree Structure - {layout_type.title()} Layout",
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
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🌳 Tree Visualization Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases the enhanced tree visualization features with:
    - **Connected nodes and edges** - Visual representation of tree structure
    - **Multiple layout algorithms** - Different ways to arrange the tree
    - **Interactive features** - Node selection and tree exploration
    - **Color coding** - Different colors for split nodes and leaf nodes
    """)
    
    # Generate sample data
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Generation")
    
    data_type = st.selectbox(
        "Data Distribution:",
        ["Uniform", "Normal", "Clustered"],
        help="Choose data distribution type"
    )
    
    data_count = st.slider("Number of Data Points:", 10, 100, 30)
    st.caption("💡 Keep data points under 100 for optimal tree visualization")
    
    if data_type == "Uniform":
        data = DataGenerator.generate_uniform(0, 100, data_count)
    elif data_type == "Normal":
        data = DataGenerator.generate_normal(50, 15, data_count)
    else:
        data = DataGenerator.generate_clusters([20, 50, 80], 8, data_count)
    
    st.write(f"Generated {len(data)} data points: {data[:10]}{'...' if len(data) > 10 else ''}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Build ANNOY tree
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.markdown("### 🌲 Tree Construction")
    
    num_trees = st.slider("Number of Trees:", 1, 5, 1)
    
    annoy = AnnoyTree(num_trees)
    annoy.build_trees(data)
    
    st.write(f"Built {len(annoy.trees)} ANNOY tree(s)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tree visualization
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.markdown("### 🎨 Tree Visualization")
    
    if annoy.trees:
        tree_index = st.selectbox(
            "Select Tree:",
            range(len(annoy.trees)),
            format_func=lambda x: f"Tree {x+1}"
        )
        
        tree_structure = annoy.get_tree_structure(tree_index)
        
        if tree_structure:
            # Layout selection
            layout_type = st.selectbox(
                "Layout Algorithm:",
                ["hierarchical", "spring", "circular"],
                format_func=lambda x: x.title(),
                help="Choose how to arrange the tree nodes"
            )
            
            # Create visualization
            nodes, edges = create_tree_visualization(tree_structure, layout_type)
            
            if nodes:
                # Display tree
                fig = display_tree_visualization(nodes, edges, layout_type)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tree statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Nodes", len(nodes))
                with col2:
                    leaf_nodes = [n for n in nodes if n['type'] == 'leaf']
                    st.metric("Leaf Nodes", len(leaf_nodes))
                with col3:
                    split_nodes = [n for n in nodes if n['type'] == 'split']
                    st.metric("Split Nodes", len(split_nodes))
                with col4:
                    st.metric("Edges", len(edges))
                
                # Interactive features
                st.markdown("#### 🔍 Interactive Features")
                
                # Node selection
                selected_node = st.selectbox(
                    "Select a node to view details:",
                    options=nodes,
                    format_func=lambda x: x['label']
                )
                
                if selected_node:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Node Information:**")
                        st.markdown(f"""
                        - **Type:** {selected_node['type'].title()}
                        - **Value:** {selected_node['value']}
                        - **Position:** ({selected_node['x']}, {selected_node['y']})
                        """)
                    
                    with col2:
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
                
                # Tree exploration
                st.markdown("#### 🌿 Tree Exploration")
                
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
                
                # Path to root
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Layout comparison
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.markdown("### 🔄 Layout Comparison")
    
    if annoy.trees and tree_structure:
        st.markdown("Compare different layout algorithms for the same tree:")
        
        layouts = ["hierarchical", "spring", "circular"]
        cols = st.columns(len(layouts))
        
        for i, layout in enumerate(layouts):
            with cols[i]:
                st.markdown(f"**{layout.title()} Layout**")
                nodes, edges = create_tree_visualization(tree_structure, layout)
                if nodes:
                    fig = display_tree_visualization(nodes, edges, layout)
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 