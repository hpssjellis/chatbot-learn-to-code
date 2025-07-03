import * as tf from '@tensorflow/tfjs';

// Assume N_LOOKBACK is defined, e.g., for a time series model looking back 10 steps
const N_LOOKBACK = 10;

const model = tf.sequential();

// Your initial LSTM layer, set up for sequence input
model.add(tf.layers.lstm({
  units: 64,             // Number of LSTM units (internal memory cells)
  inputShape: [N_LOOKBACK, 1], // Input shape: [sequence_length, number_of_features]
  returnSequences: false // Output only the last hidden state (for sequence-to-one prediction)
}));

// --- Existing Dense Layer ---
model.add(tf.layers.dense({
  units: 32,             // Number of neurons in this hidden layer
  activation: 'relu'     // Common activation function for hidden layers
}));

// --- New Layer Examples (with sensible parameters for this context) ---

// 1. tf.layers.dropout()
// Prevents overfitting by randomly setting a fraction of input units to 0.
// Often placed after a dense layer to regularize its outputs.
model.add(tf.layers.dropout({
  rate: 0.2              // Drop out 20% of the neurons from the previous layer's output
}));

// 2. tf.layers.batchNormalization()
// Normalizes the activations of the previous layer. Helps stabilize and accelerate training.
// Can be placed after dense layers or convolutional layers.
model.add(tf.layers.batchNormalization());

// 3. tf.layers.dense() - Another example, you already have one but good to show another usage
// Another fully connected layer to process features further.
model.add(tf.layers.dense({
  units: 16,             // Fewer units than the previous dense layer, further condensing information
  activation: 'relu'
}));

// --- Alternative or Additional Layer for different model architectures ---

// 4. tf.layers.gru()
// A Gated Recurrent Unit layer, often used as an alternative to LSTM.
// If used, it would typically replace the initial LSTM, or be used in stacked recurrent layers.
// Example as a replacement for the first LSTM, if you wanted to try GRU instead:
/*
const gruModel = tf.sequential();
gruModel.add(tf.layers.gru({
  units: 64,
  inputShape: [N_LOOKBACK, 1],
  returnSequences: false
}));
gruModel.add(tf.layers.dense({ units: 32, activation: 'relu' }));
gruModel.add(tf.layers.dense({ units: 1 }));
*/
// Or if you wanted to stack them (less common for simple sequence-to-one):
/*
model.add(tf.layers.lstm({
  units: 64,
  inputShape: [N_LOOKBACK, 1],
  returnSequences: true // Must be true if stacking recurrent layers
}));
model.add(tf.layers.gru({
  units: 32,
  returnSequences: false // Only the final output needed for the next dense layer
}));
*/

// 5. tf.layers.repeatVector()
// Repeats the input n times. Useful if you want to take a single vector output (e.g., from an encoder)
// and repeat it to be processed by a decoder in a sequence-to-sequence model.
// Not directly applicable after `returnSequences: false` in your current setup, but conceptually:
/*
// If the first LSTM returned a single vector and you wanted to expand it:
model.add(tf.layers.lstm({ units: 64, inputShape: [N_LOOKBACK, 1], returnSequences: false }));
model.add(tf.layers.repeatVector({ n: N_LOOKBACK })); // Repeats the 64-unit vector N_LOOKBACK times
model.add(tf.layers.lstm({ units: 32, returnSequences: true })); // Now process this repeated sequence
*/

// 6. tf.layers.timeDistributed()
// Applies a layer to every time step of a sequence. Useful for applying a Dense layer
// to each output of an LSTM when `returnSequences: true`.
// Example if the first LSTM returned sequences:
/*
model.add(tf.layers.lstm({ units: 64, inputShape: [N_LOOKBACK, 1], returnSequences: true }));
model.add(tf.layers.timeDistributed({
  layer: tf.layers.dense({ units: 32, activation: 'relu' }) // Apply this Dense layer to each timestep
}));
model.add(tf.layers.dense({ units: 1 })); // Final output layer
*/

// --- Layers for Image/2D Data (less common for simple time series, but good to know) ---
// These wouldn't typically fit directly after an LSTM unless you're reshaping.

// 7. tf.layers.conv2d()
// Applies 2D convolution. If your 'features' were actually a small image at each time step.
/*
model.add(tf.layers.conv2d({
  filters: 16,
  kernelSize: 3,
  activation: 'relu',
  inputShape: [28, 28, 1] // Example: For initial image processing
}));
*/

// 8. tf.layers.maxPooling2d()
// Downsamples the input. Companion to conv2d.
/*
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2]
}));
*/

// 9. tf.layers.flatten()
// Flattens the input. Essential to connect convolutional/pooling layers to dense layers.
// If you had Conv/Pooling, you'd add this before your dense layers.
/*
// ... after some conv2d and maxPooling2d layers
model.add(tf.layers.flatten()); // Flattens output for dense layers
*/

// 10. tf.layers.embedding()
// Turns positive integers (word IDs) into dense vectors. Used as the first layer in NLP.
// Not directly applicable after an LSTM, but if your input was a sequence of word IDs:
/*
const nlpModel = tf.sequential();
nlpModel.add(tf.layers.embedding({
  inputDim: 10000,       // Vocabulary size
  outputDim: 128,        // Embedding vector size
  inputLength: N_LOOKBACK // Sequence length
}));
nlpModel.add(tf.layers.lstm({
  units: 64,
  returnSequences: false
}));
nlpModel.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
*/


// --- Final Output Layer ---
// Your existing output layer for a single prediction.
model.add(tf.layers.dense({
  units: 1                // Single output neuron for a regression task
}));

// Compile the model (essential before training)
model.compile({
  optimizer: tf.train.adam(0.001), // Adam optimizer with a learning rate
  loss: 'meanSquaredError'         // Common loss for regression problems
});

model.summary();











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
