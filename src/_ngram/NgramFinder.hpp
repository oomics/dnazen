
#include <cmath>
#include <csignal>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// #include "fasta.hpp"
#include "ThreadPool.hpp"

typedef int Token_t;
typedef std::vector<Token_t> TokenSeq_t;

struct VectorHash {
    size_t operator()(const TokenSeq_t &vec) const {
        // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
        size_t seed = vec.size();
        for (auto x : vec) {
            size_t _x = (size_t)x;
            _x = ((_x >> 16) ^ _x) * 0x45d9f3b;
            _x = ((_x >> 16) ^ _x) * 0x45d9f3b;
            _x = (_x >> 16) ^ _x;
            seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

typedef std::unordered_map<TokenSeq_t, size_t, VectorHash> NgramMap_t;
typedef std::unordered_map<TokenSeq_t, size_t, VectorHash> PairMap_t;

// only detect keyboard interrupt if it defined
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#define DETECT_KEY_INTERRUPT()                                               \
    signal(SIGINT, [](int) {                                                 \
        std::cout << "Keyboard interrupt detected. Exiting..." << std::endl; \
        exit(0);                                                             \
    });
#else
#define DETECT_KEY_INTERRUPT()
#warning "Keyboard interrupt detection is not yet implemented on this platform"
#endif

// builtin likely
#define likely(x) __builtin_expect((x), 1)

/* utility functions */
template <typename T>
void filter_dict_by_freq(T &dict, size_t min_freq) {
    static_assert(std::is_same<decltype(std::declval<T>().begin()),
                               typename T::iterator>::value,
                  "T must have a begin() method returning an iterator");
    static_assert(std::is_same<decltype(std::declval<T>().end()),
                               typename T::iterator>::value,
                  "T must have an end() method returning an iterator");
    static_assert(
        std::is_same<decltype(++std::declval<typename T::iterator &>()),
                     typename T::iterator &>::value,
        "T's iterator must support increment");

    std::vector<typename T::key_type> keys_to_erase;
    for (auto it = dict.begin(); it != dict.end(); it++) {
        if (it->second < min_freq) {
            keys_to_erase.push_back(it->first);
        }
    }
    for (auto key : keys_to_erase) {
        dict.erase(key);
    }
}

void _count_token_and_pairs(const TokenSeq_t &token_seq,
                            std::unordered_map<Token_t, size_t> &token_dict,
                            PairMap_t &pair_map) {
    Token_t curr_token, prev_token;
    curr_token = token_seq[0];
    prev_token = 0;
    token_dict[curr_token]++;

    for (size_t i = 1; i < token_seq.size(); i++) {
        prev_token = curr_token;
        curr_token = token_seq[i];

        token_dict[curr_token]++;
        TokenSeq_t token_pair = {prev_token, curr_token};
        pair_map[token_pair]++;
    }

    return;
}

void _count_token_and_pairs_iterative(
    const std::vector<TokenSeq_t>::const_iterator &begin,
    const std::vector<TokenSeq_t>::const_iterator &end,
    std::unordered_map<Token_t, size_t> &token_dict, PairMap_t &pair_map) {
    for (auto it = begin; it != end; ++it) {
        _count_token_and_pairs(*it, token_dict, pair_map);
    }
}

struct NgramFinderConfig {
    size_t min_ngram_freq;
    size_t min_ngram_len;
    size_t max_ngram_len;
    double min_pmi;

    size_t min_token_count;

    int num_workers = 16;
};

class DnaNgramFinder {
   private:
    struct NgramFinderConfig config;
    std::unordered_map<Token_t, size_t> token_dict;
    NgramMap_t ngrams;
    PairMap_t pairs;
    // ThreadPool thread_pool;

    /**
     * @brief Filter the pairs using pmi.
     */
    void _filter_pairs(size_t total_num_tokens) {
        std::vector<TokenSeq_t> keys_to_erase;
        for (auto it = this->pairs.begin(); it != this->pairs.end(); it++) {
            auto pair = it->first;
            const size_t count = it->second;

            const Token_t a = pair[0];
            const Token_t b = pair[1];

            double mi =
                (log2(total_num_tokens) + log2(count) -
                 log2(this->token_dict[a]) -
                 log2(this->token_dict[b]));  // here we break the formula
                                              // into parts to avoid overflow

            if (mi < this->config.min_pmi) {
                keys_to_erase.push_back(pair);
            }
        }

        for (auto k : keys_to_erase) {
            this->pairs.erase(k);
        }
    }

    /**
     * @brief Updates the new n-gram dictionary with the frequency of n-grams
     * found in the token sequence.
     *
     * This function iterates over the given token sequence and generates
     * n-grams of varying lengths. It checks if each n-gram exists in the
     * provided n-gram dictionary. If it does, the function increments the count
     * of that n-gram in the new n-gram dictionary.
     *
     * @param token_seq The sequence of tokens to process.
     * @param token_dict A dictionary mapping tokens to their frequencies (not
     * used in this function).
     * @param ngram_dict A dictionary containing existing n-grams to check
     * against.
     * @param new_ngram_dict The dictionary to update with the frequency of new
     * n-grams found
     */
    void _get_new_ngram_by_freq(
        const TokenSeq_t &token_seq,
        const std::unordered_map<Token_t, size_t> &token_dict,
        const NgramMap_t &ngram_dict, NgramMap_t &new_ngram_dict) {
        for (size_t idx = 0; idx < token_seq.size(); idx++) {
            for (size_t ngram_len = 1; ngram_len < this->config.max_ngram_len &&
                                       idx + ngram_len <= token_seq.size();
                 ngram_len++) {
                TokenSeq_t ngram =
                    TokenSeq_t(token_seq.begin() + idx,
                               token_seq.begin() + idx + ngram_len);
                if (ngram_dict.find(ngram) == ngram_dict.end()) {
                    continue;
                }

                new_ngram_dict[ngram]++;
            }
        }
    }

   public:
    DnaNgramFinder(NgramFinderConfig config):token_dict(), ngrams(), pairs() {
        this->config = config;
    }

    std::unordered_map<Token_t, size_t> &get_words() {
        return this->token_dict;
    }
    NgramMap_t &get_ngrams() { return this->ngrams; }
    PairMap_t &get_pairs() { return this->pairs; }

    /**
     * @brief Converts the n-grams stored in the object to a list of token
     * sequences. The n-grams are sorted by their frequency in descending order
     * before being added to the list.
     *
     * @param ngram_list A reference to a vector of TokenSeq_t where the
     * n-grams' token sequences will be stored.
     *
     * @return List of ngrams. the ngram is a list with the last value being
     * it's frequency. Format: [<id1>, <id2>, ..., <idn>, <freq>]
     */
    std::vector<TokenSeq_t> &ngram_to_list(
        std::vector<TokenSeq_t> &ngram_list) {
        // Create a vector of pairs from the ngram map
        std::vector<std::pair<TokenSeq_t, size_t>> ngram_vector(
            this->ngrams.begin(), this->ngrams.end());

        // Sort the vector by the frequency (second element of the pair) in
        // descending order
        std::sort(
            ngram_vector.begin(), ngram_vector.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

        // Append the sorted n-grams to the list
        for (const auto &pair : ngram_vector) {
            TokenSeq_t tmp = pair.first;
            tmp.push_back(pair.second);
            ngram_list.push_back(tmp);
        }

        return ngram_list;
    }

    void find_ngrams_batched(const std::vector<TokenSeq_t> &token_seq_vec) {
        int num_threads = this->config.num_workers;
        ThreadPool thread_pool(num_threads);
        thread_pool.init();

        std::cout << "count token and pairs..." << std::endl;
        size_t total_num_tokens = 0;
        { 
            // make the actual number of tasks bit bigger
            int num_tasks = num_threads + 2;
            std::unordered_map<Token_t, size_t> token_dicts[num_tasks];
            PairMap_t pair_list[num_tasks];

            // Divide the work among threads
            auto chunk_size = token_seq_vec.size() / num_tasks;
            std::vector<std::future<void>> futures;

            for (int i = 0; i < num_tasks; ++i) {
                auto begin = token_seq_vec.begin() + i * chunk_size;
                auto end = (i == num_tasks - 1) ? token_seq_vec.end()
                                                : begin + chunk_size;

                futures.emplace_back(thread_pool.submit([&, begin, end, i] {
                    _count_token_and_pairs_iterative(begin, end, token_dicts[i],
                                                    pair_list[i]);
                }));
            }

            // Wait for all threads to finish
            for (auto &f : futures) {
                f.get();
            }

            // Merge the results from all tasks
            for (int i = 0; i < num_tasks; ++i) {
                for (const auto &pair : token_dicts[i]) {
                    this->token_dict[pair.first] += pair.second;
                    total_num_tokens += pair.second;
                }
                for (const auto &pair : pair_list[i]) {
                    this->pairs[pair.first] += pair.second;
                }
            }
        }

        // filter the mappings based on the minimum count
        filter_dict_by_freq(
            this->token_dict, this->config.min_token_count);
        filter_dict_by_freq<PairMap_t>(this->pairs,
                                       this->config.min_token_count);

        // filter the pairs based on pmi
        std::cout << "filter pairs.." << std::endl;
        this->_filter_pairs(total_num_tokens = total_num_tokens);

        // iterate over the token sequences again.
        // enumerate over ngram candidates
        std::cout << "iterate over token seq" << std::endl;
        for (auto &tok_seq : token_seq_vec) {
            TokenSeq_t ngram_candidate = {tok_seq[0]};
            for (size_t i = 1; i < tok_seq.size(); i++) {
                TokenSeq_t tok_pair = {tok_seq[i - 1], tok_seq[i]};
                if (this->pairs.find(tok_pair) != this->pairs.end() &&
                    ngram_candidate.size() < this->config.max_ngram_len) {
                    // the current token pair exists
                    ngram_candidate.push_back(tok_seq[i]);
                } else if (ngram_candidate.size() <
                           this->config.min_ngram_len) {
                    ngram_candidate = {tok_seq[i]};
                } else {
                    this->ngrams[ngram_candidate]++;
                    ngram_candidate = {tok_seq[i]};
                }
            }
        }

        filter_dict_by_freq<NgramMap_t>(this->ngrams,
                                        this->config.min_ngram_freq);

        // renew ngram by freq
        std::cout << "renew ngram by freq..." << std::endl;
        NgramMap_t new_ngram_dict;
        for (auto &tok_seq : token_seq_vec) {
            this->_get_new_ngram_by_freq(tok_seq, this->token_dict,
                                         this->ngrams, new_ngram_dict);
        }

        this->ngrams = new_ngram_dict;
        std::cout<< "filer dict by freq..." << std::endl;
        filter_dict_by_freq<NgramMap_t>(this->ngrams,
                                        this->config.min_ngram_freq);


        // thread_pool.shutdown();
    }
};