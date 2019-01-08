# Project 2. NER with HMM_MEMM
Common Terminologies Used: NER: Named Entity Recognition, PRF: Precision, Recall, F-scores,HMM: Hidden Markov Model, MEMM: Maximum, Entropy Markov Model, LGR: Logistic Regression, POS: Part of Speech

## Baseline

* Most frequent class classifier:
The assumption to choose this model is that different NER tags for words are not equally likely. Many words are only associated with one tag and most words inside a sentence are not name entities. This indicates that a model based on frequency will have acceptable performance although it can not encode the sequence
* Logistic regression classifier:
Name entities tend to have similar properties (e.g., capitalized, pos tags). In order to integrate these features in our sequential model, a logistic regression classifier is trained. Since the features
used in training the classifier are all extracted from the text, we think it is reasonable to use this classifier itself as a baseline model.

## Sequence tagging model
* HMM
Hidden Markov Model is used for named entity BIO tagging as a sequence model. It is probabilistic model that computes a probability distribution over a sequence of labels and selects the best label. The implementation of the algorithm is achieved with the bigram Viterbi algorithm.
* MEMM
HMM cannot incorporate more context or long distance features. It is essentially a logistic regression classifier, merely by designing a list of customized features, it is able to do well on tagging task. It models the distribution of predicted label (tag in this case) given the data (features).
The involved features including:
  1. <b>Is Capitalized.</b> Most of the named entities are proper nouns like name of person, organization, location which is usually capitalized Eg. America, Barack, W.H.O.
  2. <b>Is Start Of Sentence.</b> Previous feature looks if word is capitalized but we want to keep track of its position as well. Because, start of sentence is also capitalized. Ex. The game
  3. <b>Is Number.</b> Numbers are also named entity which will be part of MISC category in our case
  4. <b>has Hyphen Some. </b>
  5. <b>Is Noun (POS Tag).</b> We want to know is the part of speech associated with the word is noun as it is more likely to be named entity then compared to other part of speech
  6. <b>Is Proper Noun (POS Tag).</b> We want to know is the part of speech associated with the word is noun as it is more likely to be named entity then compared to other part of speech
  7. <b>Follows a Determiner.</b> Nouns usually follow Determiner. Ex. The United States of America
  8. <b>Follows an Adjective.</b> Nouns are mostly seen in sequences like Determiner-Adjective-Noun
  9. <b>Most likely NER Tag.</b>
  10. <b>Previous NER Tag as per prediction.</b>
  11. <b>Previous of Previous NER Tag.</b>
  12. <b>Normalized Word Embedding of Current Word (size: 300) [Continuous Feature]</b>
  13. <b>Normalized Word Embedding of Previous Word (size: 300) [Continuous Feature]</b>
  14. <b>Normalized Word Embedding of Next Word (size: 300) [Continuous Feature]</b>

## Result

|Experiments |Precision |Recall| F1| 
|:--------------:|:-----------:|:-----------:|:-------------:|
|Baseline Model - Most Frequent Class |0.716| 0.744| 0.690|
|Baseline Model -Logistic Regression| 0.594 |0.780 |0.675|
|HMM  |0.710| 0.744| 0.727|
|MEMM |0.866| 0.822| 0.953|

<b>Model Performance</b><br>

|Model / Category |ORG |MISC |PER |LOC|
|:--------------:|:-----------:|:-----------:|:-------------:|:-------------:|
|HMM |0.68| 0.64| 0.64 |0.84|
|MEMM |0.70 |0.73 |0.79 |0.82|

<b>MM and MEMM prediction</b>


According to the table, we can see that, when we use very few features, HMM outperforms MEMM.However, the performance of MEMM improves as more and more features are added to the model. This makes
sense because the advantage for MEMM over HMM is that it can use arbitrary features and encode as many features as possible. When there is only very little extra information about the context, MEMM loses its advantage. But when there are a lot of relevant features, MEMM eventually performs well because it has much more context information than HMM. As to the run time, HMM is faster. Features with higher weight are more important for prediction. For MEMM Hand Tuned features, We observed that continuous features for part of speech had huge impact on our development accuracy which is valid as nouns are
potential candidate for NER Tag. Also, word embeddings had very less weight assigned to its features indicating that using them only marginally improves accuracy. We observed a drop in test accuracy over validation accuracy as we have included too many features and may have overfit the model.

## Run time

Average Run Times for HMM and MEMM observed are 18.59s and 230.8s correspondingly.
