# Topic modeling of social history information in clinical notes

The goal of this project was to classify social history information in clinical notes using topic modeling. The major steps involved in this project were: data collection, data preprocessing, topic model training and  output interpretation. 

## Data description
Data used for this project were from the “Consult – History and Physical” notes on  [MTSamples.com (MTS)](http://www.mtsamples.com/). MTS is a public web repository with about 5,000 sample clinical notes. A total of 491 “Consult – History and Physical” notes from MTSamples.com (MTS) obtained in 2012 were used for this analysis. 1381 sentences containing social history information were manually extracted, annotated and used as the corpus.

## Data preprocessing
The corpus was pre-processed by removing punctuations, case, numbers and stop words before being used to create a lexicon consisting of 1844 terms. In the `lex_file.txt`, each term was associated with a numeric index. Information of each document was summarized in numeric format, with term index and its associated frequency. Term index and frequency for each document was integrated into `doc_file2.txt` corresponding to information in the corpus. All the preprocessing and data preparation work were carried out in the TextAnalysis (v0.1.0) Julia package. 

## Model training
Filtered Latent dirichlet allocation (fLDA) was implemented to the lexicon file and document file created in the previous step using the TopicModelsVB (v0.0.1) Julia package. A semantic theme was assigned to each topic based on its keyword list. For every clinical notes in the corpus, a vector of topic weights could be obtained from the model and used to assign each note to a topic with the greatest weight.
