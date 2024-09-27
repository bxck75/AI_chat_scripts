# filename: faiss_vector_store_plot.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import faiss
from sklearn.decomposition import PCA

class VectorStorePlotter:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def plot(self):
        # Get the vectors from the FAISS vector store
        vectors = self.vector_store.index.ntotal
        vectors = self.vector_store.index.reconstruct_n(0, vectors)

        # Reduce the dimensionality of the vectors to 3 using PCA
        pca = PCA(n_components=3)
        vectors = pca.fit_transform(vectors)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the vectors
        ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2])

        # Show the plot
        plt.show()
if __name__ == '__main__':
    # Create a FAISS vector store
    vector_store = faiss.IndexFlatL2(384)

    # Add some random vectors to the store
    vectors = np.random.rand(100, 384).astype('float32')
    vector_store.add(vectors)

    # Create a VectorStorePlotter object and plot the vectors
    plotter = VectorStorePlotter(vector_store)
    plotter.plot()