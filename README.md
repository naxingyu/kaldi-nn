# kaldi-nn
Extended speech recognition neural network based on Kaldi for reproducible research

# Changelog
1. Add ReLU, SoftHinge to nnet1
2. Add Pnorm, Maxout to nnet1
3. Optimize Maxout GPU kernel code (as fast as Pnorm)
4. Add LSTM w/o projection layer
