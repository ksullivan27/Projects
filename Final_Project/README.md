# NLP
CIS 5300

This is my portion of the CIS 5300 NLP final project. The tennis data and inspiration came from the following project:
- Tie-breaker: Using language models to quantify gender bias in sports journalism (2016)
- by Liye Fu, Cristian Danescu-Niculescu-Mizil, and Lillian Lee
- Cornell University
- https://www.cs.cornell.edu/~cristian/papers/gender-tennis.pdf

This project aims to analyze gender bias in tennis journalism by training language models on game commentary and comparing post-match interview question perplexities between men and women. My team decided to build on the original project's bigram model, looking at perplexity scores in trigram and LSTM models. Additionally, we conducted a topic modeling analysis and clustered question embeddings, comparing the centroid values with the vectors of the commentary words and GPT-generated tennis terms. Finally, we used GPT to predict and fine-tuned BERT and sportsBERT models to predict the gender of the question recipients.

Edit: I continued to work on the LSTM portion of the notebook, switching from a wrapped to a superior padded method of achieving uniform batch sizes. The findings did not change, but the figures will appear slightly different in the notebook and report/presentation.
