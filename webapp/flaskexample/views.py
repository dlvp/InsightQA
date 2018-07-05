from flask import Flask,render_template,request,redirect


import pandas as pd
import re
import numpy as np


import string
import gensim
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import preprocess_string,strip_tags,strip_punctuation, remove_stopwords,strip_numeric
import nltk
from nltk.corpus import stopwords


from flaskexample import app

stop_words = stopwords.words('english')
stop_words.extend(['q','w','e','t','y'\
                   ,'u', 'i', 'o', 'p', 'l', 'j', 'h', 'g', 'f', 'd', 's', 'a', \
                   'z', 'x', 'c', 'v', 'b', 'n', 'm'])

def parser(text):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation]
    tokens = preprocess_string(text, CUSTOM_FILTERS)
    tokens = re.sub(r'\b\d+(?:\.\d+)?\s+', '', ' '.join(tokens)).split()
    tokens = [x for x in tokens if not x in stop_words]
    return tokens



from scipy import spatial
def dist(a, b):
    return spatial.distance.cosine(a, b)


import pickle
with open("./flaskexample/dict_QA_id2head.pickle", "rb") as fp:   # Unpickling
    dict_id2head = pickle.load(fp)
with open("./flaskexample/dict_QA_head2thread.pickle", "rb") as fp:   # Unpickling
    dict_head2thread = pickle.load(fp)
with open("./flaskexample/w2v_dict_dsse.pickle", "rb") as fp:   # Unpickling
    w2v_dict = pickle.load(fp)
with open("./flaskexample/q2v_dict.pickle", "rb") as fp:   # Unpickling
    q2v_dict = pickle.load(fp)



def get_sentence_wv(sent):
    return sum([x for x in map(w2v_dict.get, parser(sent)) if x is not None])

def get_best_Qs(sent, n):
    v_sent = get_sentence_wv(sent)
    if isinstance(v_sent, int):
        return 0
    else:
        sorted_cosines = sorted(q2v_dict.items(), key=lambda x: dist(v_sent,x[1]))
        return [sorted_cosines[i][0] for i in range(n) ]




BS_color = "#CDC9C9"
DS_color = "#000000"
def return_color(score):
    if score>0.5:
        return DS_color
    else:
        return BS_color



df_processed = pd.read_csv('./flaskexample/insightslack_processed.csv',sep='\t')
df_processed.drop('Unnamed: 0', axis=1, inplace=True)


#app = Flask(__name__)

app.vars={}

@app.route('/')
@app.route('/home',methods=['GET','POST'])
def home():
    #nquestions=app_lulu.nquestions
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # request was a POST
        app.vars['question'] = request.form['question']

        f = open('./flaskexample/question.txt','w')
        f.write('%s\n'%(app.vars['question']))
        f.close()

        with open("./flaskexample/all_questions.txt", "a") as myfile:
            myfile.write(app.vars['question']+'\n')

    return redirect('/find_questions')


def qtag(score):
    if score<0.2:
        return '<font color=#FF4633> <b> GREAT MATCH</b></font>'
    if (score<0.3)&(score>=0.2):
        return '<font color=#FF7A33> <b>  GOOD MATCH</b></font>'
    if (score<0.35)&(score>=0.3):
        return '<font color=#FFB533> <b> DECENT MATCH</b></font>'
    if (score<0.45)&(score>=0.35):
        return '<font color=#FFE633> <b> SORT OF OK</b></font>'
    if (score>=0.45):
        return '<font color=#959594> <b> MEH</b></font>'


@app.route('/find_questions',methods=['GET'])
def find_qs():
    q = app.vars['question']
    ids_best = get_best_Qs(q, 10)
    
    if isinstance(ids_best, int):
        return render_template('error.html')
    else:
        #texts
        sq1 = df_processed[df_processed.ID == ids_best[0]].text.values[0]
        sq2 = df_processed[df_processed.ID == ids_best[1]].text.values[0]
        sq3 = df_processed[df_processed.ID == ids_best[2]].text.values[0]
        sq4 = df_processed[df_processed.ID == ids_best[3]].text.values[0]
        sq5 = df_processed[df_processed.ID == ids_best[4]].text.values[0]
        sq6 = df_processed[df_processed.ID == ids_best[5]].text.values[0]
        sq7 = df_processed[df_processed.ID == ids_best[6]].text.values[0]
        sq8 = df_processed[df_processed.ID == ids_best[7]].text.values[0]
        sq9 = df_processed[df_processed.ID == ids_best[8]].text.values[0]
        sq10 = df_processed[df_processed.ID == ids_best[9]].text.values[0]
        #scores
        ss1 = dist(get_sentence_wv(q),get_sentence_wv(sq1))
        ss2 = dist(get_sentence_wv(q),get_sentence_wv(sq2))
        ss3 = dist(get_sentence_wv(q),get_sentence_wv(sq3))
        ss4 = dist(get_sentence_wv(q),get_sentence_wv(sq4))
        ss5 = dist(get_sentence_wv(q),get_sentence_wv(sq5))
        ss6 = dist(get_sentence_wv(q),get_sentence_wv(sq6))
        ss7 = dist(get_sentence_wv(q),get_sentence_wv(sq7))
        ss8 = dist(get_sentence_wv(q),get_sentence_wv(sq8))
        ss9 = dist(get_sentence_wv(q),get_sentence_wv(sq9))
        ss10 = dist(get_sentence_wv(q),get_sentence_wv(sq10))
        return render_template('similar_questions.html',query=q,\
                               q1=sq1.decode('utf-8'),s1=qtag(ss1),\
                               q2=sq2.decode('utf-8'),s2=qtag(ss2),\
                               q3=sq3.decode('utf-8'),s3=qtag(ss3),\
                               q4=sq4.decode('utf-8'),s4=qtag(ss4),\
                               q5=sq5.decode('utf-8'),s5=qtag(ss5),\
                               q6=sq6.decode('utf-8'),s6=qtag(ss6),\
                               q7=sq7.decode('utf-8'),s7=qtag(ss7),\
                               q8=sq8.decode('utf-8'),s8=qtag(ss8),\
                               q9=sq9.decode('utf-8'),s9=qtag(ss9),\
                               q10=sq10.decode('utf-8'),s10=qtag(ss10))



@app.route('/answers/<string:page_name>/')
def render_static_answers(page_name):
    id = int(page_name[-1])-1
    q = app.vars['question']
    ids = get_best_Qs(q, 10)[id]
    vec = dict_head2thread[ids]
    q0 = '['+df_processed[df_processed.ID == vec[0]].user.values[0]+']'+': '+\
            unicode(df_processed[df_processed.ID == vec[0]].text.values[0],'utf-8')
    
    
    tagout = '</font></p>'
    blurb = ''
    for i in range(1,len(vec)):
        tagin = '<p><font color='+return_color(df_processed[df_processed.ID== vec[i]].DS_score.values[0])+'>'
        blurb =  blurb + tagin + '['+unicode(df_processed[df_processed.ID == vec[i]].user.values[0],'utf-8')+']'+': '
        blurb  = blurb + unicode(df_processed[df_processed.ID == vec[i]].text.values[0],'utf-8') + tagout
    
    #qq = df_processed[df_processed.ID == ids_best[id]].text.values[0]
    return render_template('%s.html' % page_name, question = q0, blurb = blurb)





################################
####### STATIC PART ############
################################

@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)



#if __name__ == "__main__":
#    app.run(host='0.0.0.0', debug = True)
