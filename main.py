import numpy as np
import matplotlib.pyplot as plt



def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    """


    K = centroids.shape[0]

   
    idx = np.zeros(X.shape[0], dtype=int)

    
    for i in range(X.shape[0]):
        min_dist = float('inf')
        
        for j in range(K):
           
            dist = np.sum((X[i] - centroids[j])**2)
            
           
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
        
   
    
    return idx

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    """
    
    m, n = X.shape
    
    
    centroids = np.zeros((K, n))
    
    
    for i in range (K):
        points = X[idx==i]
        centroids[i] = np.mean(points, axis = 0)
        
        
    
    
    return centroids



def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
   
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        
        idx = find_closest_centroids(X, centroids)
        
       
       
            
       
        centroids = compute_centroids(X, idx, K)

    return centroids, idx

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    """
    
   
    randidx = np.random.permutation(X.shape[0])
    
   
    centroids = X[randidx[:K]]
    
    return centroids

' Using on some sample data'

try:
    original_img = plt.imread('Profile.jpeg')
except FileNotFoundError:
    print("Error: Could not find Profile.jpeg. Check the filename and folder!")


original_img = original_img / 255.0 
X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))


K = 16
max_iters = 10

initial_centroids = kMeans_init_centroids(X_img, K)

centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)


idx = find_closest_centroids(X_img, centroids)
X_recovered = centroids[idx, :] 
X_recovered = np.reshape(X_recovered, original_img.shape) 

'Used AI to generate better visualization code for this last part'
fig, ax = plt.subplots(1, 2, figsize=(16, 8)) 
ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()

ax[1].imshow(X_recovered)
ax[1].set_title(f'Compressed with {K} colors')
ax[1].set_axis_off()

plt.show() 