// Assuming you have 'tf' imported: import * as tf from '@tensorflow/tfjs';

// --- Convolutional Layers (for image processing, etc.) ---

// tf.layers.conv2d()
// Applies a 2D convolution over inputs. Essential for Convolutional Neural Networks (CNNs).
// Parameters:
//   filters: Number of convolution filters (output channels).
//   kernelSize: Size of the convolution kernel (e.g., [3, 3] for a 3x3 kernel).
//   activation: Activation function (e.g., 'relu', 'sigmoid').
//   inputShape: Required for the first layer (e.g., [height, width, channels]).
// model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', inputShape: [28, 28, 1] }));

// tf.layers.maxPooling2d()
// Downsamples the input representation by taking the maximum value over a spatial window.
// Reduces computational cost and helps with translation invariance.
// Parameters:
//   poolSize: Size of the pooling window (e.g., [2, 2]).
//   strides: How far the pooling window moves for each step (defaults to poolSize).
// model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

// --- Regularization Layers (to prevent overfitting) ---

// tf.layers.dropout()
// Randomly sets a fraction of input units to 0 at each update during training time.
// This helps prevent overfitting by forcing the network to learn more robust features.
// Parameters:
//   rate: Fraction of the input units to drop (e.g., 0.2 for 20% dropout).
// model.add(tf.layers.dropout({ rate: 0.2 }));

// tf.layers.batchNormalization()
// Normalizes the activations of the previous layer at each batch, i.e., applies a transformation
// that maintains the mean activation close to 0 and the standard deviation close to 1.
// Helps stabilize and accelerate the training of deep neural networks.
// model.add(tf.layers.batchNormalization());

// --- Recurrent Layers (for sequential data like text or time series) ---

// tf.layers.lstm()
// Long Short-Term Memory layer. A type of recurrent neural network (RNN) layer
// that is well-suited for learning from sequential data with long-range dependencies.
// Parameters:
//   units: Dimensionality of the output space.
//   returnSequences: Whether to return the full sequence of outputs or just the last output.
// model.add(tf.layers.lstm({ units: 128, returnSequences: true }));

// tf.layers.gru()
// Gated Recurrent Unit layer. Another type of recurrent neural network layer,
// often considered a simpler alternative to LSTM, also good for sequential data.
// Parameters:
//   units: Dimensionality of the output space.
// model.add(tf.layers.gru({ units: 64 }));

// --- Embedding and Other Utility Layers ---

// tf.layers.embedding()
// Turns positive integers (indexes) into dense vectors of fixed size.
// Commonly used as the first layer in Natural Language Processing (NLP) models.
// Parameters:
//   inputDim: Size of the vocabulary (maximum integer index + 1).
//   outputDim: Dimensionality of the dense embedding.
//   inputLength: Length of input sequences (optional, but good for fixed-size inputs).
// model.add(tf.layers.embedding({ inputDim: 10000, outputDim: 128, inputLength: 50 }));

// tf.layers.flatten()
// Flattens the input. This is often used to transition from convolutional layers
// (which output 2D or 3D tensors) to dense layers (which expect 1D vectors).
// model.add(tf.layers.flatten());

// tf.layers.concatenate() (This is a functional API concept, not a sequential layer directly added)
// Combines a list of tensors along a specified axis. Useful for merging different
// branches of a neural network or for attention mechanisms.
// Example (requires functional API):
// const inputA = tf.input({shape: [10]});
// const inputB = tf.input({shape: [20]});
// const merged = tf.layers.concatenate().apply([inputA, inputB]);
// const output = tf.layers.dense({units: 1}).apply(merged);
// const model = tf.model({inputs: [inputA, inputB], outputs: output});

// tf.layers.layerNormalization()
// Normalizes the activations of the previous layer across the features dimension.
// Similar to Batch Normalization but independent of batch size, making it suitable
// for recurrent networks and smaller batch sizes.
// model.add(tf.layers.layerNormalization());
