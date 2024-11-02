# Neural Network Classifier from Scratch

This project involves the implementation of a neural network classifier from scratch, aiming to classify data points from a synthetic spiral dataset. The primary goal was to deepen the understanding of neural network architecture, training procedures, and optimization techniques by building the model without relying on high-level libraries.

## Project Goals

- **Understand Neural Networks**: The aim was to gain insights into how neural networks operate, including the processes of forward and backward propagation, loss calculation, and weight updates.
- **Implement Core Components**: Essential building blocks of a neural network were created, such as layers, activation functions, loss functions, and optimizers, ensuring a hands-on understanding of each component.
- **Explore Different Optimizers**: Various optimization algorithms were experimented with, including Adagrad, RMSProp, and Adam, to observe their effects on training performance and convergence.
- **Incorporate Regularization Techniques**: L2 regularization and dropout were implemented to prevent overfitting and improve the model's generalization to unseen data.
- **Evaluate Model Performance**: The effectiveness of the model was assessed by tracking accuracy and loss metrics on both training and testing datasets.

## Dataset

The dataset used is a synthetic spiral dataset, generated using the `nnfs` library, consisting of 100 samples for each of the three classes. This dataset was specifically designed to challenge the classifier's ability to discern patterns.

## Model Architecture

The neural network was structured as follows:
- An input layer connected to a Dense layer with 64 neurons.
- A ReLU activation function applied after the first Dense layer to introduce non-linearity.
- An output Dense layer with 3 neurons representing the three classes.
- A softmax loss function for multi-class classification tasks.

## Training Process

The model was trained over 10,000 iterations, following these steps:
1. **Forward Pass**: The input data was processed through the network to generate predictions.
2. **Loss Calculation**: The softmax loss and accuracy were computed based on predictions compared to the actual labels.
3. **Backward Pass**: Gradients were calculated, and the weights of the model were updated using the chosen optimizer.
4. **Logging**: Accuracy and loss values were recorded every 1,000 iterations to monitor progress throughout training.

## Regularization and Dropout

To enhance the model's ability to generalize, L2 regularization was applied to the Dense layers, and dropout layers were incorporated to randomly deactivate a fraction of neurons during training.

## Results

The final model was evaluated on a separate test dataset, achieving an accuracy of approximately 97%. This result demonstrates the effectiveness of the model in classifying the synthetic spiral dataset.

## Conclusion

This project successfully demonstrates the implementation of a neural network from scratch, emphasizing the importance of understanding the underlying mechanisms and techniques to enhance performance and generalization.
