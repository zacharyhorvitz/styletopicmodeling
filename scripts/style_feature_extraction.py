
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer, AutoModelForSequenceClassification, XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification, pipeline
import torch
import numpy as np
import csv
from urllib.request import urlopen

import pysbd
from PassivePySrc import PassivePy

from tqdm import tqdm
from gram2vec import vectorizer

import time 

UNI_TAGS = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']

def extract_pos_features(x):
    emb = {tag: 0 for tag in UNI_TAGS}
    tags = [tag[1] for tag in pos_tag(word_tokenize(x), tagset = 'universal')]
    for tag in tags:
        emb[tag] += 1

    return emb

def get_word_embeddings(model):
    state_dict = model.state_dict()
    params = []
    for key in state_dict:
        if 'word_embeddings' in key:
            params.append((key,state_dict[key]))
    assert len(params) == 1, f'Found {params}'
    return params[0][1]

def load_formality_model():
    # MODEL = f"cointegrated/roberta-base-formality"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # labels= ['informal', 'formal']
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # return model, tokenizer, embeddings, labels

    tokenizer = XLMRobertaTokenizerFast.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')
    model = XLMRobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')
    model.eval()
    embeddings = get_word_embeddings(model)

    labels = ['Formal', 'Informal']
    return model, tokenizer, embeddings, labels


def load_sentiment_model(task='sentiment'):
    # Tasks:
    # emoji, emotion, hate, irony, offensive, sentiment
    # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    embeddings = get_word_embeddings(model)
    return model, tokenizer, embeddings, labels


def load_emotion_model():
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    return classifier

def load_question_model():
    tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/question-vs-statement-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/question-vs-statement-classifier")
    model.eval()
    embeddings = get_word_embeddings(model)
    labels = ['statement', 'question']
    return model, tokenizer, embeddings, labels




def to_sentences(x, sent_detector):
    return sent_detector.segment(x)

def is_passive(x, passivepy):
    return passivepy.match_text(x, full_passive=True, truncated_passive=True)['binary_truncated_passive'].any()

def process_docs(docs, device='cpu'):
    sent_detector = pysbd.Segmenter(language="en", clean=False)

    doc_features = [{'text':d} for d in docs]
    for doc in doc_features:
        doc['sentences'] = to_sentences(doc['text'], sent_detector)

    del sent_detector
    
    passivepy = PassivePy.PassivePyAnalyzer(spacy_model = "en_core_web_lg")

    for doc in tqdm(doc_features, desc='Passive voice'):
        doc['passive'] = [is_passive(s, passivepy) for s in doc['sentences']]

    del passivepy

    model, tokenizer, _, labels = load_formality_model()

    model.to(device)

    for doc in tqdm(doc_features, desc='Formality'):
        doc['formality'] = []
        for s in doc['sentences']:
            encoded_input = tokenizer(s, return_tensors='pt')
            encoded_input.to(device)
            output = model(**encoded_input)
            predicted = torch.argmax(torch.softmax(output[0], dim=-1)[0]).item()
            predicted = labels[predicted]
            doc['formality'].append(predicted)

    del model

    for doc in tqdm(doc_features, desc='POS'):
        doc['pos'] = extract_pos_features(doc['text'])

    for task in ['emoji', 'emotion', 'hate', 'irony', 'offensive', 'sentiment']:
        model, tokenizer, _, labels = load_sentiment_model(task=task)

        model.to(device)

        for doc in tqdm(doc_features, desc=task):
            doc[task] = []
            for s in doc['sentences']:
                encoded_input = tokenizer(s, return_tensors='pt')
                encoded_input.to(device)
                output = model(**encoded_input)
                predicted = torch.argmax(torch.softmax(output[0], dim=-1)[0]).item()
                predicted = labels[predicted]
                doc[task].append(predicted)

        del model

    model = load_emotion_model()
    for doc in tqdm(doc_features, desc='Emotion'):
        doc['emotion'] = {x['label']: x['score'] > 0.5 for x in model(s)[0]}
    
    del model

    model, tokenizer, _, labels = load_question_model()
    model.to(device)

    for doc in tqdm(doc_features, desc='Question'):
        doc['question'] = []
        for s in doc['sentences']:
            encoded_input = tokenizer(s, return_tensors='pt')
            encoded_input.to(device)
            output = model(**encoded_input)
            predicted = torch.argmax(torch.softmax(output[0], dim=-1)[0]).item()
            predicted = labels[predicted]
            doc['question'].append(predicted)

    del model

    print('Vectorizing with gram2vec...')
    vectorized = vectorizer.from_documents([doc['text'] for doc in doc_features])
    columns = vectorized.columns
    for i, doc in enumerate(doc_features):
        doc['gram2vec'] = {k:v for k,v in vectorized.iloc[i].to_dict().items() if not k.startswith('letters')}

    return doc_features

if __name__ == '__main__':
    sample_corpus = ['This is a sample sentence. This is another sentence.', 'The book was being read. Did you read the book?', 'LOL :)', 'Thank you sir.', 'I love you <3']
    sample_corpus = sample_corpus * 1000
    print(f'Processing {len(sample_corpus)} documents...')
    start_time = time.time()
    processed = process_docs(docs=sample_corpus)
    print(f'Time taken: {time.time() - start_time}')
    import pdb; pdb.set_trace()

    # print(processed)
