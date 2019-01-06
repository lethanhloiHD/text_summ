from flask import Flask, render_template, request
from src.Graph_base import *
app = Flask(__name__)

@app.route('/')
def init():
    text= " "
    return render_template("init.html",summary_text =(text,text))

@app.route('/action',methods= ['POST'])
def result():
    if request.method == 'POST':
        text = request.form.get('field-input')
        print(text)
        lexrank = Rank(text,
                       tfidf_option=True,
                       doc2vec_option=False,
                       word2vec_option=False,
                       autoencoder_option=False
                       )
        sum_lexrank = lexrank.summary(option_mmr=True, using_postion_score=True)
        print(sum_lexrank)
        return render_template("init.html",summary_text = (text,sum_lexrank))

if __name__ == '__main__':
   app.run(debug = True)