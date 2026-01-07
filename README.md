# Image Compression via Vector Quantization (K-Means)

This project demonstrates an unsupervised learning approach to image compression using the **K-Means Clustering** algorithm. By treating pixels as data points in a 3D color space, we can "cluster" thousands of unique colors into a representative palette of just 16 colors.



## 🧠 Technical Overview

The core objective is to minimize the **Within-Cluster Sum of Squares (WCSS)**. Each pixel $x^{(i)}$ is assigned to a centroid $\mu_j$ that minimizes the squared Euclidean distance:

$$||x^{(i)} - \mu_j||^2$$

### Key Functions Implemented:
* `find_closest_centroids`: Iterates through every pixel to find the nearest color cluster.
* `compute_centroids`: Re-calculates the center of each cluster by averaging the RGB values of all assigned pixels.
* `kMeans_init_centroids`: Implements random initialization by selecting $K$ random pixels from the image to prevent local optima.



[Image of K-Means clustering algorithm flowchart]


## 📈 Visualizing Compression

In a standard 24-bit image, each pixel requires 8 bits for Red, 8 for Green, and 8 for Blue. 
By using K-Means with $K=16$:
1. We store a **Color Palette** (16 colors $\times$ 24 bits).
2. We store the **Image Map** where each pixel is just a 4-bit index ($2^4 = 16$).

**Result:** A compression factor of approximately **6x** with minimal perceptual loss.

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/kmeans-image-compression.git](https://github.com/yourusername/kmeans-image-compression.git)
   cd kmeans-image-compression

2. Install Dependencies:

Bash

pip install numpy matplotlib
3. Run the Script: Place your image (named Profile.jpeg) in the project folder and run:

Bash

python main.py

🖼️ Sample Output
The script generates a side-by-side comparison. The "Original" image contains thousands of colors, while the "Compressed" version uses only the 16 most dominant color centers found by the algorithm.
<img width="1163" height="548" alt="image" src="https://github.com/user-attachments/assets/5128a315-2a7d-4912-83af-28f694b8a38c" />

Developed as part of a Machine Learning exploration into Unsupervised Learning.
