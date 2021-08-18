import logging
import random
import copy
import numpy as np
import torch
from torch.nn.functional import embedding

from util.log import setup_default_logging
from infer import predict
from text import Word, Text

_logger = logging.getLogger('train')
# setup_default_logging()

class GreedyAttack():
    def __init__(self, pop_size: int, max_iters: int, embedding_model, sa_model, tokenizer, tagger) -> None:
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.embedding_model = embedding_model
        self.sa_model = sa_model
        self.tokenizer = tokenizer
        self.tagger = tagger

    def getNReplacement(self, text_obj: Text, word_obj: Word, k: int=100) -> list:
        candidate_pos = ['VV', 'VA', 'NNG', 'NNP', 'MAG']
        replace_texts = []
        idx = word_obj.idx
        word = word_obj.word
        start_idx = word_obj.start_idx
        end_idx = word_obj.end_idx
        pos = [token[1] for token in word_obj.tokens]
        nearest_neighbors = [neighbor for neighbor in self.embedding_model.get_nearest_neighbors(word, k=k) if not len(neighbor[1]) > len(word)*5]
        nearest_words = [neighbor[1] for neighbor in nearest_neighbors]
        nearest_distances = [neighbor[0] for neighbor in nearest_neighbors]

        for near_word, dist in zip(nearest_words, nearest_distances):
            replace_text = ''.join([text_obj.text[:start_idx], near_word, text_obj.text[end_idx+1:]])
            replace_text_obj = Text(replace_text, self.tagger)
            replace_pos = [token[1] for token in replace_text_obj.words[idx].tokens]
            # if pos == replace_pos:
            #     replace_texts.append((replace_text_obj, dist))
            if pos[0] == 'MAG' and replace_pos[0] == 'MAG':
                replace_texts.append((replace_text_obj, dist))
            elif replace_pos[0] in candidate_pos and pos[-1] == replace_pos[-1]:
                replace_texts.append((replace_text_obj, dist))
        
        if len(replace_texts) > 0:
            return replace_texts
        else:
            return None

    def attack(self, x_orig: str, adv_target: int) -> Text:
        x = Text(x_orig, self.tagger)
        x_len = len(x.words)

        candidates = x.getReplaceCandidates()
        candidates_replaces = dict()
        removes = []
        for cd in candidates:
            best_replaces = self.getNReplacement(x, cd)
            if best_replaces is not None:
                best_words = [(best[0][cd.idx], best[1]) for best in best_replaces]
                candidates_replaces[cd.word] = best_words
            else:
                removes.append(cd)
        for r in removes:
            candidates.remove(r)

        for i in range(self.max_iters):
            replace_text = x.words
            random_candidates = random.sample(candidates, random.randint(1, len(candidates)))
            for r_cd in random_candidates:
                distances = [rpl[1] for rpl in candidates_replaces[r_cd.word]]
                norm_distances = [d/sum(distances) for d in distances]
                replaces = [rpl[0] for rpl in candidates_replaces[r_cd.word]]
                random_replaces = np.random.choice(replaces, size=1, p=norm_distances)[0]
                replace_text[r_cd.idx] = random_replaces
            final_replace = ' '.join([word.word for word in replace_text])
            preds, scores = predict([final_replace], self.tokenizer, self.sa_model) # logits
            adv_scores = torch.stack(scores, dim=0)[:, adv_target]
            top_rank_idx = torch.argmax(adv_scores)
            if preds[top_rank_idx] == adv_target:
                adv_x = Text(final_replace, self.tagger)
                changes = (len(candidates), np.sum([x_word.word != adv_x_word.word for x_word, adv_x_word in zip(x, adv_x)]))
                return adv_x, float(adv_scores[top_rank_idx]), changes

class GeneticAttack():
    def __init__(self, pop_size: int, max_iters: int, embedding_model, sa_model, tokenizer, tagger) -> None:
        self.org_pop_size = pop_size
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.embedding_model = embedding_model
        self.sa_model = sa_model
        self.tokenizer = tokenizer
        self.tagger = tagger

    def generatePopulation(self, text_obj: Text) -> list:
        pop = []
        candidates = text_obj.getReplaceCandidates()
        # _logger.info(f'total {len(candidates)} candidates : {[cd.word for cd in candidates]}')
        if len(candidates) < self.pop_size:
            for cd in candidates:
                best_replaces = self.getNReplacement(text_obj, cd)
                if best_replaces is not None:
                    # _logger.info(f'{cd.word} has {len(best_replaces)} replaces')
                    pop.extend(best_replaces)
            pop = sorted(pop, key=lambda x: x[1])[:self.pop_size]
            # distances = [p[1] for p in pop]
            # norm_distances = [dist/sum(distances) for dist in distances]
            # pop_idx = np.random.choice(len(pop), size=self.pop_size, p=norm_distances)
            # pop = [pop[idx] for idx in pop_idx]
        else:
            extra = 0
            rand_candidates = random.sample(candidates, self.pop_size)
            for cd in rand_candidates:
                best_replaces = self.getNReplacement(text_obj, cd)
                if best_replaces is not None:
                    pop.extend(best_replaces[:extra+1])
                    extra = 0
                else:
                    extra += 1
        return pop

    def getNReplacement(self, text_obj: Text, word_obj: Word, k: int = 100) -> list:
        candidate_pos = ['VV', 'VA', 'NNG', 'NNP', 'MAG']
        replace_texts = []
        idx = word_obj.idx
        word = word_obj.word
        start_idx = word_obj.start_idx
        end_idx = word_obj.end_idx
        pos = [token[1] for token in word_obj.tokens]
        nearest_neighbors = [neighbor for neighbor in self.embedding_model.get_nearest_neighbors(word, k=k) if not len(neighbor[1]) > len(word)*5]
        nearest_words = [neighbor[1] for neighbor in nearest_neighbors]
        nearest_distances = [neighbor[0] for neighbor in nearest_neighbors]

        for near_word, dist in zip(nearest_words, nearest_distances):
            replace_text = ''.join([text_obj.text[:start_idx], near_word, text_obj.text[end_idx+1:]])
            replace_text_obj = Text(replace_text, self.tagger)
            replace_pos = [token[1] for token in replace_text_obj.words[idx].tokens]
            # if pos == replace_pos:
            #     replace_texts.append((replace_text_obj, dist))
            if pos[0] == 'MAG' and replace_pos[0] == 'MAG':
                replace_texts.append((replace_text_obj, dist))
            elif replace_pos[0] in candidate_pos and pos[-1] == replace_pos[-1]:
                replace_texts.append((replace_text_obj, dist))
        
        if len(replace_texts) > 0:
            return replace_texts
        else:
            return None

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

class PSOAttack():
    def __init__(self, pop_size: int, max_iters: int, embedding_model, sa_model, tokenizer, tagger) -> None:
        self.org_pop_size = pop_size
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.embedding_model = embedding_model
        self.sa_model = sa_model
        self.tokenizer = tokenizer
        self.tagger = tagger

    def generatePopulation(self, text_obj: Text, **kwargs):
        pop = []
        n = kwargs['n'] if 'n' in kwargs else None
        candidates = text_obj.getReplaceCandidates()
        if len(candidates) < self.pop_size:
            for cd in candidates:
                best_replaces = self.getNReplacement(text_obj, cd)
                if best_replaces is not None:
                    pop.extend(best_replaces)
            # pop = sorted(pop, key=lambda x: x[1])[:pop_size]
            distances = [p[1] for p in pop]
            norm_distances = [dist/sum(distances) for dist in distances]
            pop_idx = np.random.choice(len(pop), size=self.pop_size, p=norm_distances)
            pop = [pop[idx] for idx in pop_idx]
        else:
            extra = 0
            rand_candidates = random.sample(candidates, self.pop_size)
            for cd in rand_candidates:
                best_replaces = self.getNReplacement(text_obj, cd)
                if best_replaces is not None:
                    pop.extend(best_replaces[:extra+1])
                    extra = 0
                else:
                    extra += 1
        
        if n is None:
            return pop
        else:
            return pop[:n]

    def getNReplacement(self, text_obj: Text, word_obj: Word, k=100):
        candidate_pos = ['VV', 'VA', 'NNG', 'NNP', 'MAG']
        replace_texts = []
        idx = word_obj.idx
        word = word_obj.word
        start_idx = word_obj.start_idx
        end_idx = word_obj.end_idx
        pos = [token[1] for token in word_obj.tokens]
        nearest_neighbors = [neighbor for neighbor in self.embedding_model.get_nearest_neighbors(word, k=k) if not len(neighbor[1]) > len(word)*5]
        nearest_words = [neighbor[1] for neighbor in nearest_neighbors]
        nearest_distances = [neighbor[0] for neighbor in nearest_neighbors]

        for near_word, dist in zip(nearest_words, nearest_distances):
            replace_text = ''.join([text_obj.text[:start_idx], near_word, text_obj.text[end_idx+1:]])
            replace_text_obj = Text(replace_text, self.tagger)
            replace_pos = [token[1] for token in replace_text_obj.words[idx].tokens]
            # if pos == replace_pos:
            #     replace_texts.append((replace_text_obj, dist))
            if pos[0] == 'MAG' and replace_pos[0] == 'MAG':
                replace_texts.append((replace_text_obj, dist))
            elif replace_pos[0] in candidate_pos and pos[-1] == replace_pos[-1]:
                replace_texts.append((replace_text_obj, dist))
        
        if len(replace_texts) > 0:
            return replace_texts
        else:
            return None

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
            adv_x = nearest_texts[top_rank_idx][0]
            changes = (len(x.getReplaceCandidates()), np.sum([x_word.word != adv_x_word.word for x_word, adv_x_word in zip(x, adv_x)]))
            return adv_x, float(pop_adv_scores[top_rank_idx]), changes

        V = [np.random.uniform(-3, 3) for _ in range(self.pop_size)]
        V_P = [[V[t] for _ in range(x_len)] for t in range(self.pop_size)]

        for i in range(self.max_iters):
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
                adv_x = nearest_texts[top_rank_idx][0]
                changes = (len(x.getReplaceCandidates()), np.sum([x_word.word != adv_x_word.word for x_word, adv_x_word in zip(x, adv_x)]))
                return adv_x, float(pop_adv_scores[top_rank_idx]), changes

            new_nearest_texts = []
            for pid in range(self.pop_size):
                x_new = nearest_texts[pid]
                change_ratio = self.count_change_ratio(x_new, x, x_len)
                p_change = 1 - 2*change_ratio
                if np.random.uniform() < p_change:
                    best_replace = self.generatePopulation(x_new, n=1)[0][0] # only Text Object
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
            if x_new[idx] != x[idx]:
                cnt += 1
        change_ratio = float(cnt) / float(x_len)
        return change_ratio

class PerturbBaseline():
    def __init__(self, pop_size: int, max_iters: int, embedding_model, sa_model, tokenizer, tagger) -> None:
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.embedding_model = embedding_model
        self.sa_model = sa_model
        self.tokenizer = tokenizer
        self.tagger = tagger

    def getNReplacement(self, text_obj: Text, word_obj: Word, k: int=100) -> list:
        candidate_pos = ['VV', 'VA', 'NNG', 'NNP', 'MAG']
        replace_texts = []
        idx = word_obj.idx
        word = word_obj.word
        start_idx = word_obj.start_idx
        end_idx = word_obj.end_idx
        pos = [token[1] for token in word_obj.tokens]
        nearest_neighbors = [neighbor for neighbor in self.embedding_model.get_nearest_neighbors(word, k=k) if not len(neighbor[1]) > len(word)*5]
        nearest_words = [neighbor[1] for neighbor in nearest_neighbors]
        nearest_distances = [neighbor[0] for neighbor in nearest_neighbors]

        for near_word, dist in zip(nearest_words, nearest_distances):
            replace_text = ''.join([text_obj.text[:start_idx], near_word, text_obj.text[end_idx+1:]])
            replace_text_obj = Text(replace_text, self.tagger)
            replace_pos = [token[1] for token in replace_text_obj.words[idx].tokens]
            # if pos == replace_pos:
            #     replace_texts.append((replace_text_obj, dist))
            if pos[0] == 'MAG' and replace_pos[0] == 'MAG':
                replace_texts.append((replace_text_obj, dist))
            elif replace_pos[0] in candidate_pos and pos[-1] == replace_pos[-1]:
                replace_texts.append((replace_text_obj, dist))
        
        if len(replace_texts) > 0:
            return replace_texts
        else:
            return None

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
        