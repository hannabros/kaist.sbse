
import re

class Word:
    def __init__(self, idx, word, start_idx, end_idx) -> None:
        self.idx = idx
        self.word = word
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.tokens = []
    
    def __repr__(self) -> str:
        return self.word

    def __len__(self) -> int:
        return len(self.word)

class Text:
    def __init__(self, text, tagger) -> None:
        self.org_text = text
        self.text = re.sub("[^A-Za-z0-9가-힣\s]", "", text)
        self.offset_text = self.getOffset(self.text)
        self.morphs_text = self.getMorphs(self.text, tagger)
        self.tokens_per_word = self.addPos(self.offset_text, self.morphs_text)
        self.words = [Word(idx, word, start_idx, end_idx) for idx, (word, start_idx, end_idx) in enumerate(self.offset_text)]
        for word, tokens in zip(self.words, self.tokens_per_word):
            tmp = []
            for token in tokens:
                if '+' in token[1]:
                    pos_split = token[1].split('+')
                    tmp.extend([('', pos) for pos in pos_split[:-1]])
                    tmp.append((token[0], pos_split[-1]))
                else:
                    tmp.append((token[0], token[1]))
            word.tokens = tmp

    def __getitem__(self, i):
        return self.words[i]

    def __repr__(self) -> str:
        return(self.text)

    def getOffset(self, text):
        words = text.split()
        index = text.index
        offsets = []
        append = offsets.append
        running_offset = 0
        for word in words:
            word_offset = index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            append((word, word_offset, running_offset - 1))
        return offsets
    
    def getMorphs(self, text, tagger):
        # node = self.m.parseToNode(text)
        node = tagger.parseToNode(text)
        tokens = []
        while node:
            if node.surface:
                tokens.append(f"{node.surface}//{node.feature.split(',')[0]}//{len(node.surface)}")
            node = node.next
        return tokens

    def addPos(self, offset_text, morphs_text):
        cur_pos = 0
        tokens_per_word = []
        for offsets in offset_text:
            tmp = []
            length = offsets[2]-offsets[1]+1
            for token in morphs_text[cur_pos:]:
                if length > 0:
                    l = int(token.split('//')[2])
                    length -= l
                    tmp.append((token.split('//')[0], token.split('//')[1]))
                    cur_pos += 1
                else:
                    break
            tokens_per_word.append(tmp)
        return tokens_per_word

    def getReplaceCandidates(self):
        candidate_pos = ['VV', 'VA', 'NNG', 'NNP', 'MAG', 'XR']
        candidates = []
        for word in self.words:
            if word.tokens[0][1] in candidate_pos and len(word) > 1:
                candidates.append(word)
        return candidates