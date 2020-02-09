# Dimensional Sentiment Analysis Using a Regional CNN-LSTM Model
The code for paper "Dimensional Sentiment Analysis Using a Regional CNN-LSTM Model"

## Abstract
Dimensional sentiment analysis aims to
recognize continuous numerical values in
multiple dimensions such as the valencearousal
(VA) space. Compared to the categorical
approach that focuses on sentiment
classification such as binary classification
(i.e., positive and negative), the dimensional
approach can provide more fine-grained
sentiment analysis. This study proposes a
regional CNN-LSTM model consisting of
two parts: regional CNN and LSTM to predict
the VA ratings of texts. Unlike a conventional
CNN which considers a whole
text as input, the proposed regional CNN
uses an individual sentence as a region, dividing
an input text into several regions
such that the useful affective information in
each region can be extracted and weighted
according to their contribution to the VA
prediction. Such regional information is sequentially
integrated across regions using
LSTM for VA prediction. By combining the
regional CNN and LSTM, both local (regional)
information within sentences and
long-distance dependency across sentences
can be considered in the prediction process.
Experimental results show that the proposed
method outperforms lexicon-based, regression-based,
and NN-based methods proposed
in previous studies.
