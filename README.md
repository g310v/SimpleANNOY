# 🌳 ANNOY Algorithm Visualizer

A beautiful, interactive web application that demonstrates the **Approximate Nearest Neighbors Oh Yeah (ANNOY)** algorithm using integer data. This project provides an intuitive visualization of how ANNOY builds multiple random projection trees and performs fast approximate nearest neighbor searches.

## 🚀 Features

### 📊 Data Generation
- **Customizable Data Distributions**: Generate data using Uniform, Normal, or Clustered distributions
- **Flexible Parameters**: Set minimum/maximum values and number of data points
- **Real-time Visualization**: See your data distribution as a histogram

### 🌲 ANNOY Algorithm Implementation
- **Multiple Trees**: Build configurable number of ANNOY trees (1-20)
- **Interactive Tree Visualization**: Explore tree structures with connected nodes and edges
- **Multiple Layout Algorithms**: Hierarchical, Spring, and Circular layouts
- **Tree Exploration**: Node selection, filtering, and path tracing
- **Approximate Search**: Find k-nearest neighbors with adjustable search parameters

### 🎨 Interactive UI
- **Modern Design**: Clean, responsive interface built with Streamlit
- **Real-time Updates**: Generate new data and perform searches instantly
- **Visual Results**: See search results highlighted on scatter plots
- **Statistics Dashboard**: View key metrics about your data

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/annoy-visualizer.git
   cd annoy-visualizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 How to Use

### 1. Generate Data
- Choose a data distribution type (Uniform, Normal, or Clustered)
- Set the minimum and maximum values for your data range
- Adjust the number of data points (10-1000)
- Click "Generate New Data" to create a new dataset

### 2. Configure ANNOY Parameters
- Set the number of trees (more trees = better accuracy, slower speed)
- Choose how many nearest neighbors to return (k)

### 3. Perform Searches
- Enter a query value in the search box
- Click "Search Nearest Neighbors" to find similar values
- View results in both list and visual formats

### 4. Explore Tree Structure
- Select different trees from the dropdown
- View interactive tree visualizations with connected nodes and edges
- Choose from multiple layout algorithms (Hierarchical, Spring, Circular)
- Select nodes to view detailed information and relationships
- Filter nodes by type and trace paths to root
- Understand the hierarchical structure of ANNOY

## 🔬 How ANNOY Works

The ANNOY algorithm works by building multiple random projection trees:

1. **Tree Construction**: Each tree is built by recursively splitting the data using random hyperplanes (in our case, median splits for integer data)

2. **Random Sampling**: Each tree uses a random subset of the data, ensuring diversity across trees

3. **Search Process**: When searching for nearest neighbors:
   - Navigate down each tree following the split decisions
   - Collect candidate points from leaf nodes
   - Calculate distances and return the top k results

4. **Approximation**: The algorithm trades perfect accuracy for speed by using multiple trees and approximate search

## 🎯 Key Benefits

- **Fast Search**: O(log n) search time instead of O(n) for brute force
- **Scalable**: Works well with large datasets
- **Memory Efficient**: Trees can be stored compactly
- **Approximate but Accurate**: Multiple trees provide good accuracy

## 🏗️ Project Structure

```
annoy-visualizer/
├── app.py                 # Main Streamlit application
├── annoy_algorithm.py     # Core ANNOY implementation
├── demo.py               # Comprehensive demo script
├── tree_demo.py          # Tree visualization demo
├── test_annoy.py         # Unit tests and validation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── DEPLOYMENT.md         # Deployment guide
├── PROJECT_SUMMARY.md    # Project summary
└── .gitignore           # Git ignore file
```

## 🧪 Technical Details

### Algorithm Components

- **AnnoyNode**: Represents a node in the tree (leaf or internal)
- **AnnoyTree**: Manages multiple trees and search operations
- **DataGenerator**: Creates different types of random data distributions

### Key Methods

- `build_trees()`: Constructs multiple ANNOY trees from data
- `search()`: Performs k-nearest neighbor search
- `_build_single_tree()`: Recursively builds individual trees
- `_search_tree()`: Navigates a single tree during search

## 🚀 Deployment

### Local Development
```bash
# Main application
streamlit run app.py

# Tree visualization demo
streamlit run tree_demo.py

# Run comprehensive demo
python demo.py

# Run tests
python test_annoy.py
```

### GitHub Pages Deployment
1. Create a GitHub repository
2. Push your code to the repository
3. Enable GitHub Pages in repository settings
4. Set the source to your main branch

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the original ANNOY library by Spotify
- Built with Streamlit for beautiful web interfaces
- Visualizations powered by Plotly

## 📊 Example Usage

```python
from annoy_algorithm import AnnoyTree, DataGenerator

# Generate some data
data = DataGenerator.generate_uniform(0, 100, 1000)

# Build ANNOY trees
annoy = AnnoyTree(num_trees=10)
annoy.build_trees(data)

# Search for nearest neighbors
results = annoy.search(query=50, k=5)
print(f"Nearest neighbors to 50: {results}")
```

---

**Happy exploring! 🌳✨** 