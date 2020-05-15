# cmsc773-project

## Links

/fs/clip-corpora/cord19

https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks

https://github.com/josephsdavid/cord-19-tools

https://monkeylearn.com/blog/introduction-to-topic-modeling/

https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

## Setup

virtualenv -p python3 .venv

source .venv/bin/activate

pip install -r requirements.txt

pip install cord-19-tools --upgrade

## Output of Our System

For each task, we will have a set of topics for the task. These topics are generated in an unsupervised manner
by search for documents for that task and topic modeling on those documents. Not only will we have the topics
for a task, but for each topic, we will also have documents assigned to it. Finally, we will have a summary
of these documents assigned to the topic.
