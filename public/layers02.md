model.add(tf.layers.flatten({ inputShape: [N_LOOKBACK, 1] })); // This is the first 'layer' in terms of processing

// 2. First Dense Layer (replaces the LSTM's initial learning capacity)
model.add(tf.layers.dense({ units: 128, activation: 'relu' })); // Increased units from 64 to 128 for more capacity

// 3. Second Dense Layer
model.add(tf.layers.dense({ units: 64, activation: 'relu' })); // Equivalent to the original LSTM's 64 units

// 4. Third Dense Layer (replacing the original dense(32))
model.add(tf.layers.dense({ units: 32, activation: 'relu' }));

// 5. Output Dense Layer (this makes it technically the 4th *processing* dense layer
//    after the flatten, or the 5th if you count flatten as a layer for counting purposes,
//    but typically "4 layer dense" refers to the dense layers themselves)
model.add(tf.layers.dense({ units: 1 })); // Output a single prediction
