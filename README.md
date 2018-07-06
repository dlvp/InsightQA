# InsightQ&A
Insight Q&A is a web app to search for Data Science related threads in a corpus of online conversations. You can play with it at <a href="http://insightqa.xyz/">insightqa.xyz</a>

To help you answer your Data Science questions, the app searches though a database of online conversations shared by Insight (<a href="https://www.insightdatascience.com/">insightdatascience.com</a>) Alumni on a dedicated Slack channel. </br>

When you query the app with a question, InsightQ&A looks for the most similar questions asked in the past by some Insight Fellow and retrieve the associated threads for you, hopefully helping you out with your problem.

InsightQ&A allows you also to navigate through some of the most relevant data science topics to search for a thread that may interest you.

To make all of this possible, the app uses a dedicated word2vec word embedding to browse through the Slack database, and topic modelling techniques to organize the search by topic.

This repository contains the files which are necessary to build the web app, and two excerpts of the code that is used to build the machine learning engine on which InsightQ&A is based.

If you want to know more about what really happens under the hood, read <a href="https://dlvp.github.io/ML-InsightQA/">this blogpost</a> on my blog.
