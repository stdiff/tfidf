# Recommender System without collaborative filtering

This GitHub repository is for reproducing the results of the report "[Recommender System without collaborative filtering](http://stdiff.net/?recom_sys)". 

## Data

### Training Data

The labels of documents (items) are described in `evaluation.csv`. The tables of term frequency (tf), tfidf, etc. are stored in the following CSV files in a long format.

- Training-00.csv : The stop words are not removed, Porter's stemmer is not applied
- Training-0p.csv : The stop words are not removed, Porter's stemmer is applied
- Training-s0.csv : The stop words are removed, Porter's stemmer is not applied
- Training-sp.csv : The stop words are removed, Porter's stemmer is applied

### Test Data

As we have prepared four kinds of training data, we have the corresponding test data. But we only use `Test-sp.csv` in the report. 

- Test-00.csv : The stop words are not removed, Porter's stemmer is not applied
- Test-0p.csv : The stop words are not removed, Porter's stemmer is applied
- Test-s0.csv : The stop words are removed, Porter's stemmer is not applied
- Test-sp.csv : The stop words are removed, Porter's stemmer is applied

You can try other test data, after editing source codes.

## Scripts

All provided scripts are written in Python3.

### visualisation-training.py

This produces the histogram of tfidf (histogram-of-tfidf.png) and the 2-dimensional plot of labelled data by t-SNE (labelled-documents.png).

### classification.py

This script finds the best parameters of predictive models with several settings:

- Training data (Training-00.csv, Training-0p.csv, Training-s0.csv, Training-sp.csv)
- Values of vectors (tf, idf, tfidf, ntfidf)
- Algorithms (support vector machine, random forest, penalised logistic regression)

The result is stored in classification-result.csv. It takes quite long to produces the result.

### reduction-pca.py

Using PCA, we reduce the number of predictors. This scripts calculates the behavior the accuracy and scores of three predictive models and produces a line chart (reduction-pca.png).

### reduction-variance.py

Instead of PCA, we use predictors which high variance. This script produces a line chart (reduction-variance.png). The data frame which is used to produce the image is not stored.

### reduction-variance-2.py

Removing the predictors having only one non-zero value, we do the same thing as the previous script. A line chart (reduction-variance-2.png) is produced. The data frame which is used to produce the image is not stored.

### evaluation-test.py

This scripts applies the predictive models to the test set "Test-sp.csv" and stores the result on a CSV file evaluation-test.csv. **This script should be executed before scalar-product.py and mean-distance.py.**

### scalar-product.py

This scripts computes the preference vector and compute the scalar product of the preference vector and every unlabelled document. The histograms of scalar products on the training set and the test set are produced. The scalar products on the test set are added into evaluation-test.csv after they are normalised.

### mean-distance.py

This scripts calculates the scores by using the mean distances and produce a histogram of the scores. The script also adds the result into evaluation-test.csv after the scores are normalised.





result.csv
best.csv

