# Neural-networks-and-deep-learning
A curated set of Jupyter notebooks exploring feed-forward networks, RBMs, autoencoders, CNNs, and advanced optimization methods, all designed to deepen understanding of neural networks and deep learning.

## Description
This repository contains laboratory exercises for the **"Neural Networks and Deep Learning"** course at AGH University in Krakow, Department of Computer Science. It includes practical exercises to help understand and implement various deep learning techniques. The course is designed to provide a comprehensive introduction to neural networks and deep learning, covering both theoretical concepts and practical applications. 

## Table of Contents

1. **[1]_Feed_Forward.ipynb**  
   - **Topic:** Basic feed-forward architectures (fully connected layers).  
   - **Contents:**  
     - Implementation of simple Multi-Layer Perceptrons (MLP).  
     - Explanation of data flow and activation functions (Sigmoid, ReLU, etc.).  
   - **Use Case:** A great starting point for understanding how data propagates through a standard neural network.

2. **[2]_Visualization_PCA_T-SNE.ipynb**  
   - **Topic:** Data visualization and dimensionality reduction (PCA, t-SNE).  
   - **Contents:**  
     - Applying PCA for dimensionality reduction.  
     - Demonstration of t-SNE to visualize high-dimensional datasets in 2D/3D.  
   - **Use Case:** Helpful for exploring data structure and visualizing network representations.

3. **[3]_RBM_&_Contrastive_Divergence_algorithm.ipynb**  
   - **Topic:** Restricted Boltzmann Machine (RBM) and Contrastive Divergence.  
   - **Contents:**  
     - RBM architecture explanation.  
     - Basic Contrastive Divergence algorithm implementation.  
   - **Use Case:** Introduction to unsupervised learning and the foundation for deeper probabilistic models (e.g., DBN).

4. **[4]_RBM_&_Persistent_Contrastive_Divergence.ipynb**  
   - **Topic:** RBMs with Persistent Contrastive Divergence (PCD).  
   - **Contents:**  
     - Comparison between classic CD and Persistent CD.  
     - Examples demonstrating convergence differences.  
   - **Use Case:** Enhanced RBM training, especially relevant for deeper networks.

5. **[5]_RBM_with_momentum_&_DBN.ipynb**  
   - **Topic:** Momentum in RBMs and building Deep Belief Networks (DBN).  
   - **Contents:**  
     - Incorporating momentum in weight updates.  
     - Layer-wise construction of DBNs using stacked RBMs.  
   - **Use Case:** More stable and efficient training of deep generative networks.

6. **[6]_Backpropagation.ipynb**  
   - **Topic:** The Backpropagation algorithm.  
   - **Contents:**  
     - Detailed gradient computation and weight updates.  
     - Practical examples for training multi-layer networks.  
   - **Use Case:** The fundamental principle of supervised learning in neural networks.

7. **[7]_L1_L2_penalties.ipynb**  
   - **Topic:** L1 and L2 regularization.  
   - **Contents:**  
     - How weight penalties help reduce overfitting.  
     - Differences between L1 (sparsity) and L2 (weight decay).  
   - **Use Case:** Essential for improving generalization in neural networks.

8. **[8]_ReLU_&_Max-Norm_regularization.ipynb**  
   - **Topic:** The ReLU activation function and Max-Norm regularization.  
   - **Contents:**  
     - Benefits and potential drawbacks of ReLU.  
     - Implementing Max-Norm to limit the magnitude of weights.  
   - **Use Case:** Further stabilizing and improving the training process in deep neural networks.

9. **[9]_MLP_training_dropout.ipynb**  
   - **Topic:** Dropout in MLP training.  
   - **Contents:**  
     - Random “dropping” of neurons during training.  
     - Comparison of model performance with and without dropout.  
   - **Use Case:** A widely used technique to prevent overfitting in deep networks.

10. **[10]_Autoencoder.ipynb**  
    - **Topic:** Autoencoders.  
    - **Contents:**  
      - Learning to reconstruct input data.  
      - Using autoencoders for dimensionality reduction or denoising.  
    - **Use Case:** Foundation in unsupervised learning, useful for anomaly detection or feature extraction.

11. **[11]_Autoencoder_training_with_Nesterov.ipynb**  
    - **Topic:** Autoencoders + Nesterov momentum optimization.  
    - **Contents:**  
      - Implementing an autoencoder with Nesterov Accelerated Gradient (NAG).  
      - Comparison to standard momentum for convergence speed and stability.  
    - **Use Case:** More advanced optimization that can speed up or stabilize deep model training.

12. **[12]_CNN.ipynb**  
    - **Topic:** Convolutional Neural Networks (CNNs).  
    - **Contents:**  
      - Basic convolutional and pooling layer implementations.  
      - A simple CNN architecture demo.  
    - **Use Case:** Image classification, object detection, and other spatial data tasks.

