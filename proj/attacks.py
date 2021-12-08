import logging
import random
import copy
import numpy as np
import re
from jamo import h2j, j2hcj
import torch
from torch.nn.functional import embedding
from sentence_transformers import util
import math
from collections import Counter
import time

from util.log import setup_default_logging
from infer import predict
from text import Word, Text

_logger = logging.getLogger('train')
# setup_default_logging()

class Attack():
    def __init__(self, pop_size: int, max_iters: int, embedding_model, sent_model, sa_model, tokenizer, tagger) -> None:
        self.org_pop_size = pop_size
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.embedding_model = embedding_model
        self.sent_model = sent_model
        self.sa_model = sa_model
        self.tokenizer = tokenizer
        self.tagger = tagger
        self.embedding_cache = dict()

    def generatePopulation(self, text_obj: Text, k: int = 50) -> list:
        pop = []
        candidates = text_obj.getReplaceCandidates()
        # _logger.info(f'total {len(candidates)} candidates : {[cd.word for cd in candidates]}')
        if len(candidates) < self.pop_size:
            for cd in candidates:
                best_replaces = self.getNReplacement(text_obj, cd, k)
                if best_replaces is not None:
                    # _logger.info(f'{cd.word} has {len(best_replaces)} replaces')
                    pop.extend(best_replaces)
            pop = sorted(pop, key=lambda x: x[1], reverse=True)[:self.pop_size]
        else:
            extra = 0
            rand_candidates = random.sample(candidates, self.pop_size)
            for cd in rand_candidates:
                best_replaces = self.getNReplacement(text_obj, cd, k)
                if best_replaces is not None:
                    pop.extend(best_replaces[:extra+1])
                    extra = 0
                else:
                    extra += 1
        return pop

    def getNReplacement(self, text_obj: Text, word_obj: Word, k: int = 50) -> list:
        candidate_pos = ['VV', 'VA', 'NNG', 'NNP', 'MAG', 'XR']
        replace_texts = []
        # idx = word_obj.idx
        word = word_obj.word
        # start_idx = word_obj.start_idx
        # end_idx = word_obj.end_idx
        # pos = [token[1] for token in word_obj.tokens]
        if word in self.embedding_cache:
            nearest_neighbors = self.embedding_cache[word]
        else:
            caches = []
            nearest_neighbors = []
            for neighbor in self.embedding_model.get_nearest_neighbors(word, k=k):
                if self.filterNeighbor(neighbor, word):
                    nearest_neighbors.append(neighbor)
                    caches.append(neighbor)
            self.embedding_cache[word] = caches
        # nearest_neighbors = [neighbor for neighbor in self.embedding_model.get_nearest_neighbors(word, k=k) if self.filterNeighbor(neighbor, word)]
        nearest_words = [neighbor[1] for neighbor in nearest_neighbors]
        if len(nearest_words) == 0:
            return None       
        sort_texts = self.sortBySentModel(text_obj, word_obj, nearest_words)
        if len(sort_texts) > 0:
            return sort_texts
        else:
            return None

    def sortBySentModel(self, text_obj: Text, word_obj: Word, neareset_words: list):
        candidate_pos = ['VV', 'VA', 'NNG', 'NNP', 'MAG', 'XR']
        idx = word_obj.idx
        word = word_obj.word
        start_idx = word_obj.start_idx
        end_idx = word_obj.end_idx
        pos = [token[1] for token in word_obj.tokens]
        replace_texts = []
        for near_word in neareset_words:
            replace_text = ''.join([text_obj.text[:start_idx], near_word, text_obj.text[end_idx+1:]])
            replace_text_obj = Text(replace_text, self.tagger)
            replace_texts.append(replace_text_obj)

        syntactic_distances = [self.similarity_score(word, neighbor) for neighbor in neareset_words]
        sort_texts = []
        query = text_obj.text
        query_embedding = self.sent_model.encode(query)
        document_embeddings = self.sent_model.encode([text.text for text in replace_texts])

        cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        top_results = torch.topk(cos_scores, k=len(replace_texts))
        for score, idx, s_d in zip(top_results[0], top_results[1], syntactic_distances):
            sort_texts.append((replace_texts[idx], float(score.cpu().numpy())*(1-s_d)))

        return sort_texts

    @staticmethod
    def filterNeighbor(neighbor, word):
        if len(neighbor[1]) > 5 * len(word) or len(neighbor[1]) < 2:
            return False
        if len(re.sub("[^A-Za-z0-9가-힣\s]", "", neighbor[1])) == 0:
            return False
        chr = [ord(w) for w in neighbor[1]]
        if all(c > 55175 for c in chr) and all(c < 44032 for c in chr):
            return False
        else:
            return True

    @staticmethod
    def counter_cosine_similarity(c1, c2):
        terms = set(c1).union(c2)
        dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
        magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
        magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
        return dotprod / (magA * magB)
    
    @staticmethod
    def length_similarity(c1, c2):
        lenc1 = sum(c1.values())
        lenc2 = sum(c2.values())
        return min(lenc1, lenc2) / float(max(lenc1, lenc2))

    def similarity_score(self, w1, w2):
        l1 = [ch for ch in j2hcj(h2j(w1))]
        l2 = [ch for ch in j2hcj(h2j(w2))]
        c1, c2 = Counter(l1), Counter(l2)
        sim = self.length_similarity(c1, c2) * self.counter_cosine_similarity(c1, c2) 
        if sim < 0.1:
            sim = 1.0
        return sim

class GreedyAttack(Attack):

    def attack(self, x_orig: str, adv_target: int) -> Text:
        x = Text(x_orig, self.tagger)
        pop = self.generatePopulation(x)
        if len(pop) < self.pop_size:
            self.pop_size = len(pop)
        else:
            self.pop_size = self.org_pop_size
        # _logger.info(len(pop))
        for i in range(self.max_iters):
            start_time = time.time()
            nearest_texts = [p[0] for p in pop]
            nearest_distances = [p[1] for p in pop]
            pop_preds, pop_scores = predict([text.text for text in nearest_texts], self.tokenizer, self.sa_model) # logits
            adv_scores = torch.stack(pop_scores, dim=0)[:, adv_target]
            top_rank_idx = torch.argmax(adv_scores)
            
            # if top_rank is the adversarial then return
            if pop_preds[top_rank_idx] == adv_target:
                adv_x = pop[top_rank_idx][0]
                changes = (len(x.getReplaceCandidates()), np.sum([x_word.word != adv_x_word.word for x_word, adv_x_word in zip(x, adv_x)]))
                return adv_x, float(adv_scores[top_rank_idx]), changes
            # else crossover
            else:
                elite = pop[top_rank_idx]
                pop = self.generatePopulation(elite[0])
                if len(pop) < self.pop_size:
                    self.pop_size = len(pop)
                else:
                    self.pop_size = self.org_pop_size
            print(f'{i} iteration {int(time.time()-start_time)}')


class GeneticAttack(Attack):

    def crossover(self, x1_obj: Text, x2_obj: Text) -> Text:
        x_len = len(x1_obj.words)
        # x1 = x1_obj.text.split()
        # x2 = x2_obj.text.split()
        rand_pos = random.randint(0, x_len)
        x_new = x1_obj[:rand_pos] + x2_obj[rand_pos:]
        x_new = [w.word for w in x_new]
        return Text(' '.join(x_new), self.tagger)

    def attack(self, x_orig: str, adv_target: int) -> Text:
        x = Text(x_orig, self.tagger)
        pop = self.generatePopulation(x)
        if len(pop) < self.pop_size:
            self.pop_size = len(pop)
        else:
            self.pop_size = self.org_pop_size
        # _logger.info(len(pop))
        for i in range(self.max_iters):
            start_time = time.time()
            nearest_texts = [p[0] for p in pop]
            nearest_distances = [p[1] for p in pop]
            pop_preds, pop_scores = predict([text.text for text in nearest_texts], self.tokenizer, self.sa_model) # logits
            adv_scores = torch.stack(pop_scores, dim=0)[:, adv_target]
            top_rank_idx = torch.argmax(adv_scores)
            
            # if top_rank is the adversarial then return
            if pop_preds[top_rank_idx] == adv_target:
                adv_x = pop[top_rank_idx][0]
                changes = (len(x.getReplaceCandidates()), np.sum([x_word.word != adv_x_word.word for x_word, adv_x_word in zip(x, adv_x)]))
                return adv_x, float(adv_scores[top_rank_idx]), changes
            # else crossover
            else:
                elite = [pop[top_rank_idx]]
                norm_distances = [dist/sum(nearest_distances) for dist in nearest_distances]
                parent1 = np.random.choice(self.pop_size, size=self.pop_size-1, p=norm_distances)
                parent2 = np.random.choice(self.pop_size, size=self.pop_size-1, p=norm_distances)
                perturb_childs = []
                childs = [self.crossover(pop[parent1[i]][0], pop[parent2[i]][0]) for i in range(self.pop_size-1)]
                childs_distances = [np.mean([pop[parent1[i]][1], pop[parent2[i]][1]]) for i in range(self.pop_size-1)]
                for ch, dist in zip(childs, childs_distances):
                    ch_candidates = ch.getReplaceCandidates()
                    random_word = random.sample(ch_candidates, 1)[0]
                    best_replaces = self.getNReplacement(ch, random_word)
                    if best_replaces is not None:
                        perturb_childs.append(best_replaces[0])
                    else:
                        perturb_childs.append((ch, dist))
                pop = elite + perturb_childs
            print(f'{i} iteration {int(time.time()-start_time)}')

class PSOAttack(Attack):

    def attack(self, x_orig: str, adv_target: int):
        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2

        ###
        x = Text(x_orig, self.tagger)
        x_len = len(x.words)
        pop = self.generatePopulation(x)
        if len(pop) < self.pop_size:
            self.pop_size = len(pop)
        else:
            self.pop_size = self.org_pop_size
        nearest_texts = [p[0] for p in pop]
        # nearest_distances = [p[1] for p in pop]
        # print([text.text for text in nearest_texts])
        pop_preds, pop_scores = predict([text.text for text in nearest_texts], self.tokenizer, self.sa_model) # logits
        pop_adv_scores = torch.stack(pop_scores, dim=0)[:, adv_target]
        part_elites = copy.deepcopy(nearest_texts)
        part_elites_scores = copy.deepcopy(pop_adv_scores)
        top_rank_idx = torch.argmax(pop_adv_scores)
        elite = nearest_texts[top_rank_idx]
        elite_score = torch.max(pop_adv_scores)
        if pop_preds[top_rank_idx] == adv_target:
            adv_x = elite
            changes = (len(x.getReplaceCandidates()), np.sum([x_word.word != adv_x_word.word for x_word, adv_x_word in zip(x, adv_x)]))
            return adv_x, float(pop_adv_scores[top_rank_idx]), changes

        V = [np.random.uniform(-3, 3) for _ in range(self.pop_size)]
        V_P = [[V[t] for _ in range(x_len)] for t in range(self.pop_size)]

        for i in range(self.max_iters):
            start_time = time.time()
            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2
            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)

            for pid in range(self.pop_size):
                for dim in range(x_len):
                    V_P[pid][dim] = Omega * V_P[pid][dim] + (1 - Omega) * (self.equal(nearest_texts[pid][dim].word, part_elites[pid][dim].word) + self.equal(nearest_texts[pid][dim].word, elite[dim].word))
                turn_prob = [self.sigmoid(V_P[pid][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2

                if np.random.uniform() < P1:
                    nearest_texts[pid] = Text(' '.join(self.turn(part_elites[pid], nearest_texts[pid], turn_prob, x_len)), self.tagger)
                if np.random.uniform() < P2:
                    nearest_texts[pid] = Text(' '.join(self.turn(elite, nearest_texts[pid], turn_prob, x_len)), self.tagger)

            # nearest_texts = [p[0] for p in pop]
            # nearest_distances = [p[1] for p in pop]
            pop_preds, pop_scores = predict([text.text for text in nearest_texts], self.tokenizer, self.sa_model)
            pop_adv_scores = torch.stack(pop_scores, dim=0)[:, adv_target]
            top_rank_idx = torch.argmax(pop_adv_scores)
            elite = nearest_texts[top_rank_idx]
            if pop_preds[top_rank_idx] == adv_target:
                adv_x = elite
                changes = (len(x.getReplaceCandidates()), np.sum([x_word.word != adv_x_word.word for x_word, adv_x_word in zip(x, adv_x)]))
                return adv_x, float(pop_adv_scores[top_rank_idx]), changes

            new_nearest_texts = []
            for pid in range(self.pop_size):
                x_new = nearest_texts[pid]
                change_ratio = self.count_change_ratio(x_new, x, x_len)
                p_change = max(1 - 2*change_ratio, 0.0)
                if np.random.uniform() < p_change:
                    best_replace = self.generatePopulation(x_new, k=5)[0][0] # only Text Object
                    new_nearest_texts.append(best_replace)
                else:
                    new_nearest_texts.append(x_new)
            nearest_texts = new_nearest_texts
            # nearest_texts = [p[0] for p in pop]
            # nearest_distances = [p[1] for p in pop]
            pop_preds, pop_scores = predict([text.text for text in nearest_texts], self.tokenizer, self.sa_model)
            pop_adv_scores = torch.stack(pop_scores, dim=0)[:, adv_target]
            top_rank_idx = torch.argmax(pop_adv_scores)
            new_elite = new_nearest_texts[top_rank_idx]
            for pid2 in range(self.pop_size):
                if pop_adv_scores[pid2] > part_elites_scores[pid2]:
                    part_elites[pid2] = new_nearest_texts[pid2]
                    part_elites_scores[pid2] = pop_adv_scores[pid2]
            if torch.max(pop_adv_scores) > elite_score:
                elite = new_elite
                elite_score = torch.max(pop_adv_scores)
            print(f'{i} iteration {int(time.time()-start_time)}')

        return None

    @staticmethod
    def equal(a, b):
        if a == b:
            return -3
        else:
            return 3

    @staticmethod
    def sigmoid(n):
        return 1 / (1 + np.exp(-n))

    @staticmethod
    def turn(x1, x2, prob, x_len):
        x_new = []
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new.append(x1[i].word)
            else:
                x_new.append(x2[i].word)
        return x_new

    @staticmethod
    def count_change_ratio(x_new, x, x_len):
        cnt = 0
        for idx in range(x_len):
            if x_new[idx].word != x[idx].word:
                cnt += 1
        change_ratio = float(cnt) / float(x_len)
        return change_ratio

class PerturbBaseline(Attack):

    def attack(self, x_orig: str, adv_target: int):
        x = Text(x_orig, self.tagger)
        x_new = [w.word for w in x.words]

        candidates = x.getReplaceCandidates()
        for cd in candidates:
            best_replaces = self.getNReplacement(x, cd)
            if best_replaces is not None:
                best_words = [(best[0][cd.idx], best[1]) for best in best_replaces][0]
                x_new[cd.idx] = best_words[0].word
        
        x_new = ' '.join(x_new)
        pop_preds, pop_scores = predict([x_new], self.tokenizer, self.sa_model)
        adv_scores = torch.stack(pop_scores, dim=0)[:, adv_target]
        top_rank_idx = torch.argmax(adv_scores)
            
        # if top_rank is the adversarial then return
        if pop_preds[top_rank_idx] == adv_target:
            adv_x = Text(x_new, self.tagger)
            changes = (len(candidates), np.sum([x_word.word != adv_x_word.word for x_word, adv_x_word in zip(x, adv_x)]))
            return adv_x, float(adv_scores[top_rank_idx]), changes
        