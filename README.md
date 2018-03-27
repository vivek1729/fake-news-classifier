# fake-news-classifier
A DL model to classify fake news from the Liar Dataset

Automatic fake news detection is a challenging problem in deception detection, and it has tremendous real-world political and social impacts. However, statistical approaches to combating fake news has been dramatically limited by the lack of labeled benchmark datasets. Used the LIAR dataset with rich meta data to build a fake news classification model that outperformed the model mentioned in the paper (https://arxiv.org/abs/1705.00648) where the accuracy of the proposed model is around 27% on the test set.

1. There were two broad categories of features – the word statement and the meta data. A task was to have good representations for both of these features. Two options for word representations were considered – static glove embeddings taken from the Wikipedia (6 Billion token dataset) and non static embeddings initialized by the model and learned during the training time. Word embeddings (both static and non static) were 100 dimensional
2. Meta data was difficult to categorize because of the diverse information present. I tried to quantify different meta tags into discrete classes and a symbolic ‘rest’ class that allowed me to condense the values into a fixed (limited) number.
3. Developed two models for the classification task:
- LSTM based model where news text was fed to the LSTM and the output was added to a condensed representation of the Meta Data (28% test accuracy)
- CNN based model with 128 filters each of size 2,5 and 8. Meta data was added in a condensed form just like the LSTM model (30% test accuracy)
4. Extensively used Stochastic Gradient Descent and Tensorboard to visualize and develop intuition on convergence of these models. LSTM models tend to overfit easily. Dropout was used for regularization

Detailed project report can be found in the pdf file `repo.pdf`

I have included both the jupyter notebook and the cleaned python code in the repo. The cleaned python code can be found in `run.py`
