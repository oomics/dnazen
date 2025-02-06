#include "NgramFinder.hpp"

#include <iostream>
#include <vector>
#include <string>
// include time.h for measuring time
#include <ctime>

// measure the time taken to read the fasta file
int main() {
    struct NgramFinderConfig config;
    config.max_ngram_len = 5;
    config.min_ngram_len = 2;
    config.min_ngram_freq = 1;
    config.min_pmi = 1;
    config.min_token_count = 3;
    config.num_workers = 16;

    DnaNgramFinder ngram_finder(config);
    
    // make random token sequences
    std::vector<TokenSeq_t> token_seq_vec = {};
    int _seed = 1;

    for (int i=0; i<10000; i++) {
        TokenSeq_t token_seq = {1,2,3,1,2,3};
        for (int j=0; j<1000; j++) {
            _seed ^= ((i+j)<<10) + (i<<4) + j + 0xdeadbeaf + (_seed << 6) + (_seed >> 2);
            token_seq.push_back(0x00ff & _seed);
        }

        token_seq_vec.push_back(token_seq);
    }
    std::cout<<"finding ngram"<<std::endl;
    ngram_finder.find_ngrams_batched(token_seq_vec);
    std::cout<<"result:"<<std::endl;

    NgramMap_t ngram_dict = ngram_finder.get_ngrams();
    for (auto it = ngram_dict.begin(); it != ngram_dict.end(); it++) {
        std::cout << "key=";
        for (Token_t tok : it->first) {
            std::cout << tok << " ";
        }
        std::cout << "val=" << it->second;
        std::cout << std::endl;
    }

    PairMap_t pair_dict = ngram_finder.get_pairs();
    for (auto it = pair_dict.begin(); it != pair_dict.end(); it++) {
        std::cout << "pair=(";
        std::cout << std::get<0>(it->first) << " ";
        std::cout << std::get<1>(it->first) << " ";
        std::cout << ") val=" << it->second;
        std::cout << std::endl;
    }

    return 0;
}