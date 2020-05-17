# dependencies: spacy, gensim
# do: !python3 -m spacy download en
import os
import sys
import pickle
# !{sys.executable} -m spacy download en
import re, numpy as np, pandas as pd
from gensim import models

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import string

IR_RESULTS = sys.argv[1]

name = IR_RESULTS.split('.')[1].split('/')[-1].split('_')[-1]
if not os.path.exists('./outputs_final'):
    os.mkdir('./outputs_final') 
if not os.path.exists('./outputs_final/lda_models_tfidf'):
    os.mkdir('./outputs_final/lda_models_tfidf') 

    

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', \
                    'say', 'could', 'be', 'know', 'good', 'go', 'get', \
                    'do', 'done', 'try', 'many', 'some', 'nice', 'thank', \
                    'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', \
                    'make', 'want', 'seem', 'run', 'need', 'even', 'right', \
                    'line', 'even', 'also', 'may', 'take', 'come', 'introduction',\
                    'methodology', 'discussion', 'results', 'discussion', \
                    'conclusion', 'background', '0','1', '2', '3', '4', '5', '6', '7', '8', '9',\
                    'virus', 'viral', 'disease', 'viruses', 'infection', \
                    'infectious', 'health', 'vaccine'])

# matplotlib inline
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# Import Dataset
df_orig = pd.read_csv('./metadata.csv')


# Import IR results
df_ir = pd.read_csv(IR_RESULTS, delimiter=' ', \
                    header=None, \
                    names=['task', 'q', 'cord_uid','rank','num','galago'])


# Filter the df per topic
def extract_topic_df(num, query_df=df_ir, metadata=df_orig):
    topic_df_ir = query_df[query_df['task']=='T-' + str(num)].reset_index()
    topic_df_meta = metadata[metadata['cord_uid'].isin(topic_df_ir.loc[:,'cord_uid'])].reset_index()
    # topic_df_meta = pd.merge(metadata, topic_df_ir, on='cord_uid')
    topic_df_filtered = topic_df_meta[['title', 'abstract']]
    return topic_df_filtered


# nltk.download('punkt')
def sent_to_words(sentences):
    tokens = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub('[!#?,.:";]', '', sent)
        sent = re.sub("\'", "", sent)  # remove single quotes
        tokens.append(tokenizer.tokenize(str(sent)))
    return tokens 

def df2list(df):
# Convert to list
    data = df.values.tolist()
    data = [x for xs in data for x in xs]
    data_words = list(sent_to_words(data))
    return data_words


# !python3 -m spacy download en  # run in terminal once
# or do
# !conda install -c conda-forge spacy-model-en_core_web_md 
# and use nlp=spacy.load('en_core_web_sm') instead in below function.
def process_words(texts, stop_words=stop_words):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
        # Build the bigram and trigram models
    bigram = gensim.models.Phrases(texts, min_count=2, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts = [[word for word in doc if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in doc if word not in stop_words] for doc in texts]   
    return texts_out


def make_lda(corpus, id2word, NUM=4):
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=NUM, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=10,
                                            passes=10,
                                            alpha='symmetric',
                                            iterations=100,
                                            per_word_topics=True)

    return lda_model



def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def format_further(df_topic_sents_keywords):
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    #df_dominant_topic.head(10)

    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100

    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpdn = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpdn:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                                axis=0)
    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

    return sent_topics_sorteddf_mallet


def get_tfidf_corpus(bow_corpus):
    """
    :return: corpus and dictionary to pass to lda_model
    """
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf




for task in range(2):
    if task == 0:
        num_task = 15
    if task == 1:
        num_task == 10
    topic_df = extract_topic_df(task + 1)
    data_words = df2list(topic_df.astype(str))

    data_ready = process_words(data_words)  # processed Text Data!
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]
    corpus_tfidf = get_tfidf_corpus(corpus)

    # do_topic_modelling(corpus, "./outputs", task)
    lda_model = make_lda(corpus, id2word, num_task)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

    sent_topics_sorteddf_mallet = format_further(df_topic_sents_keywords)

    # filepath = output_dir + '/topics_task_' + str(task+1) + '.csv'

    sent_topics_sorteddf_mallet.to_csv('./outputs_final/topics_task_' + str(task+1) + '.csv', index=False)

    # save results
    lda_file = "./outputs_final/lda_models_tfidf/lda_model_"+str(task+1)
    lda_model.save(lda_file)

    with open('./outputs_final/lda_models_tfidf/corpus_'+str(task+1)+'.pickle','wb') as f:
        pickle.dump(corpus,f)
