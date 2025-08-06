#!/usr/bin/env python3
"""
Demo script for the ANNOY Algorithm Visualizer
This script demonstrates various use cases and capabilities of the ANNOY algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from annoy_algorithm import AnnoyTree, DataGenerator

def demo_basic_functionality():
    """Demonstrate basic ANNOY functionality"""
    print("🌳 ANNOY Algorithm Demo")
    print("=" * 50)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    data = DataGenerator.generate_uniform(0, 100, 200)
    print(f"   Generated {len(data)} data points in range [0, 100]")
    print(f"   Data statistics: min={min(data)}, max={max(data)}, mean={np.mean(data):.1f}")
    
    # Build ANNOY trees
    print("\n2. Building ANNOY trees...")
    start_time = time.time()
    annoy = AnnoyTree(num_trees=5)
    annoy.build_trees(data)
    build_time = time.time() - start_time
    print(f"   Built {len(annoy.trees)} trees in {build_time:.3f} seconds")
    
    # Perform searches
    print("\n3. Performing nearest neighbor searches...")
    queries = [25, 50, 75]
    
    for query in queries:
        start_time = time.time()
        results = annoy.search(query, k=5)
        search_time = time.time() - start_time
        
        print(f"\n   Query: {query}")
        print(f"   Search time: {search_time:.4f} seconds")
        print("   Results:")
        for i, (value, distance) in enumerate(results):
            print(f"     {i+1}. Value: {value}, Distance: {distance}")

def demo_different_distributions():
    """Demonstrate ANNOY with different data distributions"""
    print("\n\n📊 Different Data Distributions Demo")
    print("=" * 50)
    
    distributions = [
        ("Uniform", DataGenerator.generate_uniform(0, 100, 300)),
        ("Normal", DataGenerator.generate_normal(50, 15, 300)),
        ("Clustered", DataGenerator.generate_clusters([20, 50, 80], 8, 300))
    ]
    
    for name, data in distributions:
        print(f"\n{name} Distribution:")
        print(f"  Data points: {len(data)}")
        print(f"  Range: [{min(data)}, {max(data)}]")
        print(f"  Mean: {np.mean(data):.1f}")
        print(f"  Std: {np.std(data):.1f}")
        
        # Build trees and search
        annoy = AnnoyTree(num_trees=3)
        annoy.build_trees(data)
        
        query = np.mean(data)
        results = annoy.search(int(query), k=3)
        print(f"  Nearest to {int(query)}: {[r[0] for r in results]}")

def demo_performance_comparison():
    """Compare ANNOY performance with different parameters"""
    print("\n\n⚡ Performance Comparison Demo")
    print("=" * 50)
    
    # Generate larger dataset
    data = DataGenerator.generate_uniform(0, 1000, 1000)
    print(f"Dataset: {len(data)} points")
    
    # Test different numbers of trees
    tree_counts = [1, 3, 5, 10]
    query = 500
    
    print("\nPerformance with different tree counts:")
    print("Trees | Build Time | Search Time | Results")
    print("-" * 45)
    
    for num_trees in tree_counts:
        # Build trees
        start_time = time.time()
        annoy = AnnoyTree(num_trees=num_trees)
        annoy.build_trees(data)
        build_time = time.time() - start_time
        
        # Search
        start_time = time.time()
        results = annoy.search(query, k=5)
        search_time = time.time() - start_time
        
        print(f"{num_trees:5d} | {build_time:9.3f}s | {search_time:10.4f}s | {len(results)}")

def demo_accuracy_analysis():
    """Analyze ANNOY accuracy vs brute force"""
    print("\n\n🎯 Accuracy Analysis Demo")
    print("=" * 50)
    
    # Generate test data
    data = DataGenerator.generate_uniform(0, 100, 500)
    query = 50
    
    # Brute force search (ground truth)
    distances = [(x, abs(x - query)) for x in data]
    distances.sort(key=lambda x: x[1])
    true_neighbors = distances[:5]
    
    print(f"Query: {query}")
    print(f"True nearest neighbors: {[x[0] for x in true_neighbors]}")
    print(f"True distances: {[x[1] for x in true_neighbors]}")
    
    # ANNOY search
    annoy = AnnoyTree(num_trees=10)
    annoy.build_trees(data)
    annoy_results = annoy.search(query, k=5)
    
    print(f"ANNOY results: {[x[0] for x in annoy_results]}")
    print(f"ANNOY distances: {[x[1] for x in annoy_results]}")
    
    # Calculate accuracy
    true_values = set(x[0] for x in true_neighbors)
    annoy_values = set(x[0] for x in annoy_results)
    overlap = len(true_values.intersection(annoy_values))
    accuracy = overlap / len(true_values) * 100
    
    print(f"Accuracy: {accuracy:.1f}% ({overlap}/{len(true_values)} correct)")

def demo_tree_visualization():
    """Demonstrate tree structure visualization"""
    print("\n\n🌲 Tree Structure Demo")
    print("=" * 50)
    
    # Generate small dataset for clear visualization
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print(f"Data: {data}")
    
    # Build single tree
    annoy = AnnoyTree(num_trees=1)
    annoy.build_trees(data)
    
    # Get tree structure
    tree_structure = annoy.get_tree_structure(0)
    
    print("\nTree Structure:")
    print_tree_visual(tree_structure)

def print_tree_visual(node, level=0, prefix=""):
    """Print tree structure in a visual format"""
    if node['is_leaf']:
        print(f"{prefix}🍃 {node['value']}")
    else:
        print(f"{prefix}🌿 Split: {node['split_value']:.1f}")
        if node['children']:
            print(f"{prefix}├── Left:")
            print_tree_visual(node['children'][0], level + 1, prefix + "│   ")
            print(f"{prefix}└── Right:")
            print_tree_visual(node['children'][1], level + 1, prefix + "    ")

def main():
    """Run all demos"""
    try:
        demo_basic_functionality()
        demo_different_distributions()
        demo_performance_comparison()
        demo_accuracy_analysis()
        demo_tree_visualization()
        
        print("\n\n🎉 Demo completed successfully!")
        print("\nTo run the interactive web app:")
        print("streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 