#include <iostream>
#include <vector>
#include <string>
// include time.h for measuring time
#include <ctime>

#include "FreqNgramFinder.hpp"

// measure the time taken to read the fasta file
int main() {
    struct FreqNgramFinderConfig config;
    config.max_ngram_len = 5;
    config.min_ngram_len = 2;
    config.num_workers = 16;

    FreqNgramFinder ngram_finder(config);
    
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

    return 0;
}