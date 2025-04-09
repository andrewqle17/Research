from VBESegmenter import VBESegmenter
from build_dictionary import read_dictionary
import similarity as sim
import preprocess as pre
import pandas as pd
import cProfile
import os
import pstats

def main():
    os.chdir("/home/andrewqle17/.Research/data")

    data = pd.read_csv("main.csv",
                    header=0,
                    index_col=0)

    os.chdir("/home/andrewqle17/.Research/scripts")

    dicts = {i: read_dictionary(f"{i}gram_dict.txt") for i in range(1, 11)}

    segmenter = VBESegmenter(dicts=dicts)

    domains = data["Domain_Name"].str[:-4]

    vbe_results = domains.apply(segmenter.segment, limits=len(segmenter.freq_dicts)-1)
    data['vbesegment'] = vbe_results
    data['vbesegment'] = data['vbesegment'].apply(pre.combine_segments)

    methods = ["Manual1", "Manual2", "ChatGPT_4o", "wordsegment", 'vbesegment']

    metrics = [sim.calc_exact_match]
    sim.calc_similarity_matrices(metrics = metrics, segmentation_methods=methods, data=data)

cProfile.run('main()', 'profile_output.txt')

p = pstats.Stats('profile_output.txt')
p.sort_stats('cumulative').print_stats(10)