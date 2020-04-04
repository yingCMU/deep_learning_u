# Attention

## Encoders and Decoders
- sequence to sequence models, using encoder & decoder
- The encoder and decoder do not have to be RNNs; they can be CNNs too!
 In computer vision, we can use this kind of encoder-decoder model to generate words or captions for an input image or even to generate an image from a sequence of input words. For now know that we can input an image into a CNN (encoder) and generate a descriptive caption for that image using an LSTM (decoder).

- encoder gives all vectors to decoder, decoder learns a context matrix to decide on the attentions (attentions can also be used to learn the order in different language translation)

- What's a more reasonable embedding size for a real-world application? 200-300


## scoring functions

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
