import string
from collections import defaultdict
from functools import lru_cache

import numpy as np
from build_dictionary import read_dictionary


class VBESegmenter:
    def __init__(self, dicts={}):
        self.freq_dicts = dicts
        self.max_length = len(dicts)
        self.charset = list(dicts[1].keys()) if 1 in dicts else []
        
        self.vbe_dicts = {'forwards': {}, 'backwards': {}}
        self.norm_constants = {'forwards': {}, 'backwards': {}}
        
        for key, dictionary in dicts.items():
            if key == self.max_length:
                continue
            ngrams = list(dictionary.keys())
            self.vbe_dicts['forwards'][key] = self.bulk_calc_vbe(ngrams, 'forwards')
            self.vbe_dicts['backwards'][key] = self.bulk_calc_vbe(ngrams, 'backwards')
            values = np.fromiter(dictionary.values(), dtype=np.float64)
            self.norm_constants['forwards'][key] = (values.mean(), values.std())
            self.norm_constants['backwards'][key] = (values.mean(), values.std())

    def bulk_calc_vbe(self, ngrams, mode):
        if not ngrams:
            return {}
        
        results = {}
        be_cache = {}
        
        for ngram in ngrams:
            if mode == 'forwards':
                context = ngram
                prev_context = ngram[:-1]
            else:
                context = ngram
                prev_context = ngram[1:]
            
            be = be_cache.get(context, self.calc_be(context, mode))
            be_prev = be_cache.get(prev_context, self.calc_be(prev_context, mode))
            results[ngram] = be - be_prev
            be_cache[context] = be
            if prev_context:
                be_cache[prev_context] = be_prev
                
        return results

    @lru_cache(maxsize=1024) 
    def calc_be(self, context, mode='forwards'):
        length = len(context)
        if length + 1 not in self.freq_dicts:
            return 0.0
        
        dictionary = self.freq_dicts[length + 1]
        total = 0.0
        counts = {}
        

        if mode == 'forwards':
            prefix = context
            for char in self.charset:
                key = prefix + char
                if key in dictionary:
                    counts[key] = dictionary[key]
                    total += dictionary[key]
        else:  # backwards
            suffix = context
            for char in self.charset:
                key = char + suffix
                if key in dictionary:
                    counts[key] = dictionary[key]
                    total += dictionary[key]
        
        if total == 0:
            return 0.0
        

        probs = np.array([count / total for count in counts.values()])
        return -np.sum(probs * np.log(probs + 1e-10))  # Add small constant to avoid log(0)

    @lru_cache(maxsize=1024)
    def calc_nvbe(self, context, mode='forwards'):
        length = len(context)
        if length not in self.norm_constants[mode]:
            return 0.0
        mean, std = self.norm_constants[mode][length]
        vbe = self.vbe_dicts[mode][length].get(context, 0.0)
        return (vbe - mean) / (std + 1e-10)  # Avoid division by zero

    def autonomy_function(self, context):
        return np.min([self.calc_nvbe(context, 'forwards'), self.calc_nvbe(context, 'backwards')])

    def segment(self, text, limits):
        cache = {}

        def search(text, prev='*'):
            if text in cache:
                return cache[text]
            
            if not text:
                return 0.0, []
            
            best_score = float('-inf')
            best_split = []
            
            limit = min(len(text), limits)
            for pos in range(1, limit + 1):
                prefix, suffix = text[:pos], text[pos:]
                prefix_score = self.autonomy_function(prefix) * len(prefix)
                suffix_score, suffix_words = search(suffix, prefix)
                total_score = prefix_score + suffix_score
                if total_score > best_score:
                    print(f"Prefix: {prefix}, Suffix: {suffix}, Score: {total_score}")
                    best_score = total_score
                    best_split = [prefix] + suffix_words
            
            result = (best_score, best_split)
            cache[text] = result
            return result
        
        _, words = search(text)
        return words

def main():
    dicts = {i: read_dictionary(f'{i}gram_dict.txt') 
            for i in range(1, 11)}
    segmenter = VBESegmenter(dicts)
    print('Instantiation done')
    print(segmenter.calc_be('', mode='forwards'), segmenter.calc_be('', mode='backwards'))
    print(segmenter.calc_be('a', mode='forwards'), segmenter.calc_be('a', mode='backwards'))
    print(segmenter.norm_constants['forwards'][1], segmenter.norm_constants['backwards'][1])
    print(segmenter.vbe_dicts['forwards'][1].get('a', 0.0), segmenter.vbe_dicts['backwards'][1].get('a', 0.0))
    print(segmenter.calc_nvbe('a', mode='forwards'))
    print(segmenter.calc_nvbe('a', mode='backwards'))
    print(segmenter.autonomy_function('a'))

if __name__ == '__main__':
    main()