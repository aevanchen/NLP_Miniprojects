# Project 3. Sequence_model_attention
## Debugging 
Given a script, find four bugs in the script relating to model file in Pytorch. 
The details of bugs are in the reports file : NLP_project3.doc
## Seq2Seq
3 toy tasks are experimented for sequence model
* Reverse Sequence
* Copy Sequence
* Sort Sequence
Sequence input are strings of numbers.

Two attention machinism are implemented in this project including (1) Additive attention and (2) Multiplicative attention. The difference between these two attention can refer to: [Attention](http://ruder.io/deep-learning-nlp-best-practices/).

## Result
Runtime (n is the sequence length)

|Approach	|Copy|	Reverse|	Sort|
|:--------------:|:-----------:|:-----------:|:-------------:|
|Neural Models	|O(n)|	O(n)	|O(n)|
|Classical Algorithms|	O(n)	|O(n) |Mergesort O(nlog(n))|


|Task	|With Teacher Forcing	|Without Teacher Forcing|
|:--------------:|:-----------:|:-----------:|
|Copy	|0.8275	|0.5216|
|Reverse|	0.9338|0.9457|
|Sort	|0.4227|	0.4431|

<b>Test Accuracy for models without attention on given test set with and without teacher forcing</b>

|Task	|Without attention	|With Additive Attention|With Multiplicative Attention|
|:--------------:|:-----------:|:-----------:|:-----------:|
|Copy	|0.8273	|0.9915 |1.0|
|Reverse|	0.9337	|0.9373 |0.9373|
|Sort	|0.4227	|0.434	|0.4484|

<b>Test Accuracy results for models with and without attention on given test set </b>


●	When we extended the sequence length from 10 to 20, the text accuracy drops from 1 to 0.75. It is proved that the sequence model does not perform well on long sequence. For a relative large vocabulary and a longer input sequence to model, it is seen that the attention performs a lot better than without. It is because it can encode more ‘focused’ dependency on the far away input vectors to learn and represent their relationship in the learning phase. We can see similar behavior in Experiment 2 and 3 due to same reason.
●	The multiplicative attention outperforms the additive attention. It has been studied that the additive is better theoretically for high dimension inputs like word embedding. One of reasons to observe a worse performance of additive attention might be that we only trained 5 epochs for the model. Since there are 3 weight matrices to optimize, the training time to the optimum solution will take longer time.

## Asymptotic Performance
●	For the sequence models without attention, the cost is relatively cheap. The encoder is linearly dependent on the input length therefore is O(n). The decoder produces the same length of output thus the time for decoding is also O(n). Other cost like generating embedding, softmax cross entropy, could be considered as constant time operation. The overall big O without attention is O(n).
●	When we add attention, encoding of input sequences remain O(n), the run time differs in the decoding phase. Now, we need a context vector in each time step. The context vector is computed by running n times of computing attention weights for each encoding states. The followed softmax normalization step and summation of n vectors could be considered as constant run time. Therefore, the complexity of decoding phase becomes O(n2) and the total running time becomes O(n)+O(n2) which is same as O(n2).
●	For additive attention and multiplicative attention, they both have the same asymptotic run time which is O(n2). Yet, their practical run times differ. The multiplicative attention is faster and more space efficient since it only has simple matrix multiplication and one weight matrix to optimize. In the other hand, additive attention involves multiplication and addition and needs to store 3 parameters, making it slower and less space efficient.

