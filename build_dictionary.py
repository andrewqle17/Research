from collections import defaultdict

from nltk.util import ngrams

# Takes a file of domain names separated by \n characters
def build_dictionary(text_path, n=1, mode='continuous'):
    ngram_dict = defaultdict(int)
    output_name = f'{n}gram_dict.txt'
    if mode=='continuous':
        with open(text_path) as f:
            text = f.read().replace('\n', '')
        for ngram in ngrams(list(text), n, pad_left=True, pad_right=True, left_pad_symbol='*', right_pad_symbol='*'):
            ngram_dict[''.join(ngram)] += 1
    if mode=='separated':
        with open(text_path) as f:
            lines = f.readlines()
            for line in lines:
                grams = ngrams(list(line.strip()), n, pad_left=False, pad_right=False)
                for ngram in grams:
                    ngram_dict[''.join(ngram)] += 1
    ngram_dict = {key: value for key, value in sorted(ngram_dict.items(), key=lambda item: item[1], reverse=True)}
    total_ngrams = sum(ngram_dict.values())
    curr_ngrams = 0
    with open(output_name, mode='w', encoding='utf-8') as output:
        for key in ngram_dict:
            output.write(f'{key}: {ngram_dict[key]}\n')
            curr_ngrams += ngram_dict[key]
            if curr_ngrams > total_ngrams:
                break
    return ngram_dict
            

def read_dictionary(dict_path):
    ngram_dict = defaultdict(int)
    with open(dict_path) as f:
        for line in f:
            line = line.strip()
            key, value = line.split(": ", 1)
            ngram_dict[str(key)] = int(value)
    print(f'{dict_path} read')
    return ngram_dict
    
def main():
    for i in range(10):
        build_dictionary("domain_text_1m.txt", n=i+1, mode='separated')

if __name__ == '__main__':
    main()

    