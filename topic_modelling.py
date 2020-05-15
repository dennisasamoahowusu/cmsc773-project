# dependencies: spacy, gensim
# do: !python3 -m spacy download en

# !{sys.executable} -m spacy download en
import pandas as pd
import re

# Gensim
import gensim
import gensim.corpora as corpora
from gensim import models
import logging
import warnings
from nltk.tokenize import RegexpTokenizer
from pprint import pprint

tokenizer = RegexpTokenizer(r'\w+')

# Plot data

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
                    'conclusion', 'background', '0','1', '2', '3', '4', '5', '6', '7', '8', '9'])

# matplotlib inline
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# Import Dataset
df_orig = pd.read_csv('./metadata.csv')


# Import IR results
df_ir = pd.read_csv('./galago_data/results_bm25_2000.csv', delimiter=' ', \
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
        # sent = re.sub('\d\Z', ' ', sent) # remove isolated digits
        # sent = re.sub(r'\[.*?\]|\(.*?\)|\W', ' ', sent) # remove brackets etc
        #tokens.append(word_tokenize(str(sent)))
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
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
        # Build the bigram and trigram models
    bigram = gensim.models.Phrases(texts, min_count=2, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts = [[word for word in doc if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    # texts_out = []
    # nlp = spacy.load('en', disable=['parser', 'ner'])
    # for sent in texts:
    #     doc = nlp(" ".join(sent)) 
    #     texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # # remove stopwords once more after lemmatization
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


def do_topic_modelling(corpus, output_dir, task):
    lda_model = make_lda(corpus, id2word)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

    sent_topics_sorteddf_mallet = format_further(df_topic_sents_keywords)

    filepath = output_dir + '/topics_task_' + str(task) + '.csv'

    sent_topics_sorteddf_mallet.to_csv(filepath, index=False)

    # PLOTTING
    # import pyLDAvis.gensim
    # pyLDAvis.enable_notebook()

    # vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    # pyLDAvis.show(vis)


for task in range(len(df_ir['task'].unique())):
    topic_df = extract_topic_df(task + 1)
    data_words = df2list(topic_df.astype(str))

    data_ready = process_words(data_words)  # processed Text Data!
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]
    corpus_tfidf = get_tfidf_corpus(corpus)

    do_topic_modelling(corpus, "./outputs", task)
    do_topic_modelling(corpus_tfidf, "./tfidf_outputs", task)



# print(merged_df.info())
# print(merged_df.shape)
# print()
# print(df_ir.info())
# print(df_ir.shape)
# print(df_ir.tail(20))

# merged_df contains the subset of the data that is contained in the IR results

# for task in range(len(df_ir['task'].unique())):
#
#     # topic_df is df for articles for a particular task
#     topic_df = extract_topic_df(merged_df, task+1)
#     print(topic_df.head(1))
#
#     # data_words is a list of the words comprising the title and abstract for each row in the df
#     data_words = df2list(topic_df.astype(str))
#     pprint(data_words[:2])
#
#     # data_ready is data_words processed - length can be different because stop words are removed and some words
#     # are joined into n-grams
#     data_ready = process_words(data_words)  # processed Text Data!
#     print()
#     print("Data Ready: ")
#     pprint(data_ready[:2])
#
#     # Create Dictionary
#     id2word = corpora.Dictionary(data_ready)
#     print(type(id2word))
#
#     # Create Corpus: Term Document Frequency
#     corpus = [id2word.doc2bow(text) for text in data_ready]
#     pprint(corpus[:2])
#
#     lda_model = make_lda(corpus, id2word)
#
#     df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)
#     print(df_topic_sents_keywords.head())
#     print(df_topic_sents_keywords.info())
#
#     sent_topics_sorteddf_mallet = format_further(df_topic_sents_keywords)
#     print(sent_topics_sorteddf_mallet.head())
#     print(sent_topics_sorteddf_mallet.info())
#
#     sent_topics_sorteddf_mallet.to_csv('./outputs/topics_task_' + str(task) + '.csv', index=False)


    # PLOTTING
    # import pyLDAvis.gensim
    # pyLDAvis.enable_notebook()
    #
    # vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    # pyLDAvis.show(vis)