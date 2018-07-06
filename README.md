# InsightQ&A
Insight Q&A is a web app to search for Data Science related threads in a corpus of online conversations. You can play with it at <a href="http://insightqa.xyz/">insightqa.xyz</a>

To help answer your Data Science question, the app looks though a database of online conversations shared by Insight (<a href="https://www.insightdatascience.com/">insightdatascience.com</a>) Alumni on a dedicated Slack channel. </br>

When you query the app with a question, InsightQ&A search for the most similar questions that have been asked already by some Insight Fellow and retrieve the associated thread for you, hopefully helping you out with your problem.

InsightQ&A also allows you to navigate the most important data science topics to search for a thread that may interest you.

To reach its goal, the app uses a dedicated word2vec word embedding to browe through the online database, and topic modelling techniques to organize the search by topic.

This repository contains
