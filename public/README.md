

Some models with a few issues?


```
mlModel.add(tf.layers.lstm({ units: 128, inputShape: [N_LOOKBACK, 1], returnSequences: true }));
mlModel.add(tf.layers.dropout({ rate: 0.2 }));
mlModel.add(tf.layers.lstm({ units: 64, returnSequences: false }));
mlModel.add(tf.layers.dense({ units: 32, activation: 'relu' }));
mlModel.add(tf.layers.dense({ units: 1 }));
```

