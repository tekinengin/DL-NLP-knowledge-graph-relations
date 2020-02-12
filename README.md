# knowledge-graph-relations

The goal of work is to train deep learning models to determine knowledge graph relations (in this case, according to the Freebase schema) that are invoked in user utterances to a conversational system. You will be given a training set of utterances paired with a set of relations, that you can use to train multi-layer perceptron (MLP) models. Here is an example utterance from the dataset:

Show me movies directed by Woody Allen recently.
There are two relations that are invoked by this utterance:

movie.directed_by
movie.initialreleasedate
The task is to train deep learning models and output the associated set of relations when given a new utterance.
