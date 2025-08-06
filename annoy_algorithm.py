import numpy as np
import random
from typing import List, Tuple, Optional
import math

class AnnoyNode:
    """Node in the ANNOY tree"""
    def __init__(self, value: int, left=None, right=None, is_leaf=True):
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.split_value = None
        self.split_dimension = None

class AnnoyTree:
    """ANNOY tree implementation for integer data"""
    
    def __init__(self, num_trees: int = 10):
        self.num_trees = num_trees
        self.trees = []
        self.data = []
        
    def build_trees(self, data: List[int]):
        """Build multiple ANNOY trees from the data"""
        self.data = data
        self.trees = []
        
        for _ in range(self.num_trees):
            # Randomly sample data points for this tree
            tree_data = random.sample(data, min(len(data), max(1, len(data) // 2)))
            tree = self._build_single_tree(tree_data)
            self.trees.append(tree)
    
    def _build_single_tree(self, data: List[int]) -> AnnoyNode:
        """Build a single ANNOY tree"""
        if len(data) == 1:
            return AnnoyNode(data[0])
        
        if len(data) == 2:
            root = AnnoyNode(None, is_leaf=False)
            root.split_value = (data[0] + data[1]) / 2
            root.left = AnnoyNode(data[0])
            root.right = AnnoyNode(data[1])
            return root
        
        # For integer data, we'll use the value itself as the split dimension
        # In a real implementation, this would be a random hyperplane
        split_value = np.median(data)
        
        left_data = [x for x in data if x <= split_value]
        right_data = [x for x in data if x > split_value]
        
        # Ensure we have data in both branches and prevent infinite recursion
        if not left_data:
            left_data = [min(data)]
        if not right_data:
            right_data = [max(data)]
        
        # If splitting doesn't reduce the data size, create a leaf node
        if len(left_data) == len(data) or len(right_data) == len(data):
            # Create a leaf node with the median value
            return AnnoyNode(int(split_value))
        
        root = AnnoyNode(None, is_leaf=False)
        root.split_value = split_value
        
        root.left = self._build_single_tree(left_data)
        root.right = self._build_single_tree(right_data)
        
        return root
    
    def search(self, query: int, k: int = 5) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors"""
        candidates = set()
        
        # Search in all trees
        for tree in self.trees:
            tree_candidates = self._search_tree(tree, query)
            candidates.update(tree_candidates)
        
        # Calculate distances and return top k
        distances = [(x, abs(x - query)) for x in candidates]
        distances.sort(key=lambda x: x[1])
        
        return distances[:k]
    
    def _search_tree(self, node: AnnoyNode, query: int, depth: int = 0) -> List[int]:
        """Search a single tree"""
        if node.is_leaf:
            return [node.value]
        
        # Navigate down the tree
        if query <= node.split_value:
            candidates = self._search_tree(node.left, query, depth + 1)
            # Also check the other branch if we're not too deep
            if depth < 3:  # Limit backtracking depth
                candidates.extend(self._search_tree(node.right, query, depth + 1))
        else:
            candidates = self._search_tree(node.right, query, depth + 1)
            # Also check the other branch if we're not too deep
            if depth < 3:  # Limit backtracking depth
                candidates.extend(self._search_tree(node.left, query, depth + 1))
        
        return candidates
    
    def get_tree_structure(self, tree_index: int = 0) -> dict:
        """Get the structure of a tree for visualization"""
        if tree_index >= len(self.trees):
            return {}
        
        return self._node_to_dict(self.trees[tree_index])
    
    def _node_to_dict(self, node: AnnoyNode) -> dict:
        """Convert a tree node to a dictionary for visualization"""
        if node.is_leaf:
            return {
                'value': node.value,
                'is_leaf': True,
                'children': []
            }
        else:
            return {
                'value': node.split_value,
                'is_leaf': False,
                'split_value': node.split_value,
                'children': [
                    self._node_to_dict(node.left),
                    self._node_to_dict(node.right)
                ]
            }

class DataGenerator:
    """Generate random integer data for testing"""
    
    @staticmethod
    def generate_uniform(min_val: int, max_val: int, count: int) -> List[int]:
        """Generate uniform random integers"""
        return list(np.random.randint(min_val, max_val + 1, count))
    
    @staticmethod
    def generate_normal(mean: int, std: int, count: int) -> List[int]:
        """Generate normally distributed integers"""
        return list(np.random.normal(mean, std, count).astype(int))
    
    @staticmethod
    def generate_clusters(centers: List[int], std: int, count: int) -> List[int]:
        """Generate clustered data around specified centers"""
        data = []
        points_per_cluster = count // len(centers)
        
        for center in centers:
            cluster_data = np.random.normal(center, std, points_per_cluster).astype(int)
            data.extend(cluster_data)
        
        # Add remaining points randomly
        remaining = count - len(data)
        if remaining > 0:
            data.extend(np.random.choice(centers, remaining))
        
        return data 