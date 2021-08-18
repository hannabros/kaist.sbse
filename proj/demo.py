
import os, sys
import fasttext
import csv
import MeCab
import random
import numpy as np
from tqdm import tqdm
import pickle
import logging
random.seed(1234)

import torch
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from CustomModel import BERTClassifier
from util.log import setup_default_logging

from text import Text, Word
from infer import predict
from attacks import GreedyAttack, GeneticAttack, PSOAttack, PerturbBaseline

_logger = logging.getLogger('train')

def save(result, result_path, algorithm, pop_size, sample_size):
    file_name = os.path.join(result_path, f'{algorithm}_p{pop_size}_s{sample_size}2.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(result, f)
    _logger.info(f'saved to {file_name}')

def main(algorithm):
    ### argument
    data_path = './data/rating_test_spell.pkl'
    result_path = './result/'
    pop_size = 50
    max_iters = 50
    sample_size = 100
    ###

    _logger.info('loading test data')
    # Load Pickle Data
    with open(data_path, 'rb') as f:
        d = pickle.load(f)
        spell_documents = d['documents']
        targets = d['targets']
        target_scores = d['target_scores']
    _logger.info(f'finished loading {len(spell_documents)} data')

    _logger.info('loading embedding model')
    # load embedding model
    embedding_model = fasttext.load_model('/home/ubuntu/workspace/kaist.sbse/proj/data/cc.ko.300.bin')

    _logger.info('loading tagger')
    # load Mecab
    tagger = MeCab.Tagger('-d /home/ubuntu/workspace/kaist.sbse/mecab-ko-dic-2.1.1-20180720')

    # load sentiment model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    _logger.info('loading tokenizer & sentiment model')
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    sa_model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
    checkpoint = torch.load('./model/pytorch_model.bin')['model_state_dict']
    sa_model.load_state_dict(checkpoint)

    _logger.info('filtering test data')
    # filter and make test_set
    test_set = [(d, t) for d, t in zip(spell_documents, targets) if len(d.split()) >= 5 and len(d.split()) < 30]
    test_set = random.sample(test_set, sample_size)

    result = []
    if algorithm == 'greedy':
        attack = GreedyAttack(pop_size=pop_size, max_iters=max_iters, embedding_model=embedding_model, sa_model=sa_model, tokenizer=tok, tagger=tagger)
    elif algorithm == 'genetic':
        attack = GeneticAttack(pop_size=pop_size, max_iters=max_iters, embedding_model=embedding_model, sa_model=sa_model, tokenizer=tok, tagger=tagger)
    elif algorithm == 'pso':
        attack = PSOAttack(pop_size=pop_size, max_iters=max_iters, embedding_model=embedding_model, sa_model=sa_model, tokenizer=tok, tagger=tagger)
    elif algorithm == 'base':
        attack = PerturbBaseline(pop_size=pop_size, max_iters=max_iters, embedding_model=embedding_model, sa_model=sa_model, tokenizer=tok, tagger=tagger)
    
    cnt = 0
    error_idx = []
    for i, t in enumerate(tqdm(test_set[10:])):
        try:
            x_orig = t[0]
            adv_target = 0 if t[1] == 1 else 1
            adv_result = attack.attack(x_orig, adv_target)
            if adv_result is not None:
                x_adv = adv_result[0]
                adv_score = adv_result[1]
                changes = adv_result[2]
                result.append((x_orig, x_adv.text, adv_score, changes))
                cnt += 1
                save(result, result_path, algorithm, pop_size, sample_size)
                _logger.info(f'succeeded {cnt}')
        except:
            error_idx.append(i)
    
    return result, error_idx

if __name__ == "__main__":
    setup_default_logging()

    algs = ['greedy', 'genetic', 'pso', 'base']
    alg = sys.argv[1]
    if alg not in algs:
        _logger.info('wrong algorithm name')
        sys.exit(1)

    main(alg)