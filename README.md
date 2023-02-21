# CovidReview

The training and testing can be used as follow: 


```<sh>

for value in digitalepidemiologylab/covid-twitter-bert-v2 dmis-lab/biobert-v1.1 roberta-base roberta-large microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
do
    python ~/model_tpu.py $value
done

```
