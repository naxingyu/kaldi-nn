# kaldi-nn
Extended speech recognition neural network based on Kaldi for reproducible research

# Changelog
1. Add ReLU, SoftHinge to nnet1
2. Add Pnorm, Maxout to nnet1
3. Optimize Maxout GPU kernel code (as fast as Pnorm)
4. Add LSTM w/o projection layer
5. Add egs/hkust for reproducing the results
6. Add ReLU result
7. Add Convolution and Maxpooling components, and test code
8. Add example for training ConvNet in nnet2
9. Add LSTM projected component in nnet2
10. Add per-utterance training binary
11. Add per-utterance options for egs-operation binaries