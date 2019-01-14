# AUTOMATIC TEXT SUMMARIZATION for single document Vietnamese

In my prpject ,  I use unsupervise model train in data on the site : soha.com, dantri.com, kenh14.com, ...

I using Doc2vec, Word2vec, LDA, Autoencoder are the unsuperivse model for representation sentence to vectorizer.
After, I use Pagerank and K-mean for select sentence from graph or cluster. 
All sentence selected will be throught MMR or plMMR for select sentence for create summary of document begin input.

For requestment :
   - python >= 3.5
   - module using in file requestment.txt

Run :
   - Input : data/data_demo/input.txt
   - Output : data/data_demo/output.txt
   - Build_model : select one option for build model : python model_build.py

 Run (simple) :
                 python run_graph.py
          or     python run_cluster.py

 Run (template): input and output will show in the templates :
                 python run_templates.py
    
