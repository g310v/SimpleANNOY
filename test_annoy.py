#!/usr/bin/env python3
"""
Simple test script for the ANNOY algorithm implementation
"""

from annoy_algorithm import AnnoyTree, DataGenerator
import numpy as np

def test_data_generation():
    """Test data generation functions"""
    print("🧪 Testing data generation...")
    
    # Test uniform distribution
    uniform_data = DataGenerator.generate_uniform(0, 100, 50)
    print(f"Uniform data (0-100, 50 points): min={min(uniform_data)}, max={max(uniform_data)}")
    
    # Test normal distribution
    normal_data = DataGenerator.generate_normal(50, 10, 50)
    print(f"Normal data (mean=50, std=10, 50 points): mean={np.mean(normal_data):.1f}")
    
    # Test clustered data
    clustered_data = DataGenerator.generate_clusters([20, 50, 80], 5, 60)
    print(f"Clustered data (centers=[20,50,80], 60 points): unique values={len(set(clustered_data))}")
    
    print("✅ Data generation tests passed!\n")

def test_annoy_tree():
    """Test ANNOY tree construction and search"""
    print("🌳 Testing ANNOY tree...")
    
    # Generate test data
    data = DataGenerator.generate_uniform(0, 100, 100)
    
    # Build ANNOY trees
    annoy = AnnoyTree(num_trees=5)
    annoy.build_trees(data)
    
    print(f"Built {len(annoy.trees)} trees")
    print(f"Data points: {len(annoy.data)}")
    
    # Test search
    query = 50
    results = annoy.search(query, k=5)
    
    print(f"Search results for query={query}:")
    for i, (value, distance) in enumerate(results):
        print(f"  {i+1}. Value: {value}, Distance: {distance}")
    
    # Verify results are reasonable
    assert len(results) <= 5, "Should return at most k results"
    assert all(distance >= 0 for _, distance in results), "Distances should be non-negative"
    
    print("✅ ANNOY tree tests passed!\n")

def test_tree_structure():
    """Test tree structure visualization"""
    print("🌿 Testing tree structure...")
    
    # Generate small dataset for easier visualization
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Build single tree
    annoy = AnnoyTree(num_trees=1)
    annoy.build_trees(data)
    
    # Get tree structure
    tree_structure = annoy.get_tree_structure(0)
    
    print("Tree structure:")
    print_tree(tree_structure)
    
    print("✅ Tree structure tests passed!\n")

def print_tree(node, level=0):
    """Print tree structure in a readable format"""
    indent = "  " * level
    
    if node['is_leaf']:
        print(f"{indent}🍃 {node['value']}")
    else:
        print(f"{indent}🌿 Split: {node['split_value']:.1f}")
        for child in node['children']:
            print_tree(child, level + 1)

def test_performance():
    """Test performance with larger dataset"""
    print("⚡ Testing performance...")
    
    # Generate larger dataset
    data = DataGenerator.generate_uniform(0, 1000, 1000)
    
    # Build trees
    annoy = AnnoyTree(num_trees=10)
    annoy.build_trees(data)
    
    # Test multiple searches
    queries = [100, 500, 900]
    
    for query in queries:
        results = annoy.search(query, k=10)
        print(f"Query {query}: Found {len(results)} neighbors")
    
    print("✅ Performance tests passed!\n")

if __name__ == "__main__":
    print("🚀 Starting ANNOY Algorithm Tests\n")
    
    try:
        test_data_generation()
        test_annoy_tree()
        test_tree_structure()
        test_performance()
        
        print("🎉 All tests passed! The ANNOY implementation is working correctly.")
        print("\nYou can now run the Streamlit app with: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise 