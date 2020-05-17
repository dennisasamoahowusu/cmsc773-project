# pip install selenium
# pip install bokeh
# conda install -c conda-forge firefox geckodriver

import os
import sys
import pickle
import pandas as pd
import numpy as np
import gensim

# PLOT - WORDCLOUD
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

# PLOT - TOP TOPICS PER DOC
from matplotlib.ticker import FuncFormatter


# PLOT - CLUSTERING CHART
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import export_png, save

NUM_DOCS = sys.argv[1]


if not os.path.exists('./figures/cluster_charts/' + str(NUM_DOCS)):
    os.mkdir('./figures/cluster_charts/' + str(NUM_DOCS)) 
if not os.path.exists('./figures/wordclouds/' + str(NUM_DOCS)):
    os.mkdir('./figures/wordclouds/' + str(NUM_DOCS)) 
if not os.path.exists('./figures/num_docs/' + str(NUM_DOCS)):
    os.mkdir('./figures/num_docs/' + str(NUM_DOCS)) 


def load_lda_model(task_num, name=NUM_DOCS):
    model_path = "./lda_models/" + "lda_models_" + NUM_DOCS + "/lda_model_"+str(task_num)
    lda_model = gensim.models.ldamodel.LdaModel.load(model_path)
    return lda_model


####################################
#WORDCLOUD PLOTTING OF TOP N WORDS#
####################################
def wordcloud_plotting(lda_model):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=STOPWORDS,
                    background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: cols[i],
                    prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig('./figures/wordclouds/' + NUM_DOCS + '/task_'+str(task)+'.jpg')


#############################
# CLUSTERING CHART PLOTTING #
#############################

def plot_cluster_chart(lda_model, corpus):
    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    n_topics = 4
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
                plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    # save(plot, filename='./figures/cluster_charts/task_'+str(task)+'.html',\
    #             title='Task '+str(task))
    export_png(plot, filename='./figures/cluster_charts/' + NUM_DOCS + '/task_'+str(task)+'.png')


# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

def dominant_topic_df(dominant_topics):
    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

    return df_dominant_topic_in_each_doc

def topic_weightage(topic_percentages):
    # Total Topic Distribution by actual weight
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

    return df_topic_weightage_by_doc

def top3words_df(lda_model):
    # Top 3 Keywords for each Topic
    topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                    for j, (topic, wt) in enumerate(topics) if j < 3]

    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0,inplace=True)
    
    return df_top3words

def plot_topics_documents(lda_model, corpus):
    
    dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            
    df_dominant_topic_in_each_doc = dominant_topic_df(dominant_topics)
    df_topic_weightage_by_doc = topic_weightage(topic_percentages)
    df_top3words = top3words_df(lda_model)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=120, sharey=True)

    # Topic Distribution by Dominant Topics
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
    ax1.set_ylabel('Number of Documents')
    #ax1.set_ylim(0, 2000)

    # Topic Distribution by Topic Weights
    ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
    ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
    ax2.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

    plt.savefig('./figures/num_docs/' + NUM_DOCS + '/task_'+str(task)+'.jpg')


for task in range(9):
    with open('./lda_models/lda_models_' + NUM_DOCS + '/corpus_'+str(task)+'.pickle',"rb") as f:
        corpus = pickle.load(f)
        lda_model = load_lda_model(task)
        # plot wordcloud for this task
        wordcloud_plotting(lda_model)
        # plot cluster chart
        plot_cluster_chart(lda_model, corpus)
        # plot top N topics and number of documents
        plot_topics_documents(lda_model, corpus)