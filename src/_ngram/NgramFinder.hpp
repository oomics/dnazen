
#include <cmath>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ThreadPool.hpp"

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

typedef uint32_t Token_t;
typedef std::vector<Token_t> TokenSeq_t;
typedef std::tuple<Token_t, Token_t> TokenPair_t;

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

struct TokenPairHash {
    size_t operator()(const TokenPair_t &token_pair) const {
        // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
        size_t seed = 0xdeadbeef;

        // manual loop-unroll
        size_t x = (size_t)std::get<0>(token_pair);
        size_t _x = x;
        _x = ((_x >> 16) ^ _x) * 0x45d9f3b;
        _x = ((_x >> 16) ^ _x) * 0x45d9f3b;
        _x = (_x >> 16) ^ _x;
        seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        x = (size_t)std::get<1>(token_pair);
        _x = x;
        _x = ((_x >> 16) ^ _x) * 0x45d9f3b;
        _x = ((_x >> 16) ^ _x) * 0x45d9f3b;
        _x = (_x >> 16) ^ _x;
        seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        return seed;
    }
};

typedef std::unordered_map<Token_t, uint64_t> TokenDict_t;
typedef std::unordered_map<TokenSeq_t, uint64_t, VectorHash> NgramMap_t;
typedef std::unordered_map<TokenPair_t, uint64_t, TokenPairHash> PairMap_t;

/* utility functions */
inline bool isInteger(const std::string &s) {
    if (s.empty() || (!isdigit(s[0]))) return false;

    char *p;
    strtol(s.c_str(), &p, 10);

    return (*p == 0);
}

void make_token_seq(std::string &line, TokenSeq_t &token_seq) {
    // splitting string
    size_t start = 0;
    size_t end = line.find(':');
    while (end != std::string::npos) {
        std::string token = line.substr(start, end - start);
        if (!isInteger(token)) {
            std::cerr << "Token " << token << " cannot be convertted to int."
                      << std::endl;
        }
        token_seq.push_back(std::stoul(token));

        start = end + 1;
        end = line.find(':', start);
    }

    // process the last string
    std::string token = line.substr(start);
    if (!isInteger(token)) {
        std::cout << "Token " << token << " cannot be convertted to int."
                  << std::endl;
    } else {
        token_seq.push_back(std::stoul(token));
    }

    return;
}

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
                            TokenDict_t &token_dict, PairMap_t &pair_map) {
    if (token_seq.empty()) {
        std::cout << "Warning: token_seq is empty" << std::endl;
        return;
    }

    Token_t curr_token, prev_token;
    curr_token = token_seq[0];
    prev_token = 0;
    token_dict[curr_token]++;

    for (size_t i = 1; i < token_seq.size(); i++) {
        prev_token = curr_token;
        curr_token = token_seq[i];

        token_dict[curr_token]++;
        TokenPair_t token_pair = {prev_token, curr_token};
        pair_map[token_pair]++;
    }

    return;
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
    TokenDict_t token_dict;
    NgramMap_t ngrams;
    std::mutex mutex;

    /**
     * @brief Filter the pairs using pmi.
     */
    void _filter_pairs(PairMap_t &pairs, size_t total_num_tokens) {
        std::vector<TokenPair_t> keys_to_erase;
        for (auto it = pairs.begin(); it != pairs.end(); it++) {
            auto pair = it->first;
            const size_t count = it->second;

            const Token_t a = std::get<0>(pair);
            const Token_t b = std::get<1>(pair);

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
            pairs.erase(k);
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
    void _get_new_ngram_by_freq(const TokenSeq_t &token_seq,
                                // const TokenDict_t &token_dict,
                                const NgramMap_t &ngram_dict,
                                NgramMap_t &new_ngram_dict) {
        NgramMap_t local_ngram_dict;

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

                local_ngram_dict[ngram]++;
            }
        }

        std::lock_guard<std::mutex> lock(mutex);
        for (const auto &pair : local_ngram_dict) {
            new_ngram_dict[pair.first] += pair.second;
        }
    }

    /**
     * @brief Count token and pairs frequency and store into two dicts.
     *
     * @param token_seq The sequence of token to precess.
     * @param token_dict A dictionary mapping tokens to their frequencies. THIS
     * is the dictionary we would update.
     * @param pairs A dictionary mapping token pairs to their frequencies. THIS
     * is the dictionary we would update.
     * @param mutex the lock used when updateing the above two dictionaries.
     */
    void _count_token_and_pairs2(const TokenSeq_t &token_seq,
                                 TokenDict_t &token_dict, PairMap_t &pairs,
                                 std::mutex &mutex) {
        TokenDict_t local_token_dict;
        PairMap_t local_pair_map;
        _count_token_and_pairs(token_seq, local_token_dict, local_pair_map);

#ifdef __DEBUG__
        std::cout << "_count_token_and_pairs2" << std::endl;
#endif

        std::lock_guard<std::mutex> lock(mutex);
        for (const auto &pair : local_token_dict) {
            token_dict[pair.first] += pair.second;
        }
        for (const auto &pair : local_pair_map) {
            pairs[pair.first] += pair.second;
        }
    }

   public:
    DnaNgramFinder(NgramFinderConfig config) : token_dict(), ngrams() {
        DETECT_KEY_INTERRUPT()
        this->config = config;
    }

    TokenDict_t &get_tokens() { return this->token_dict; }
    NgramMap_t &get_ngrams() { return this->ngrams; }

    /**
     * @brief Converts the n-grams stored in the object to a list of token
     * sequences. The n-grams are sorted by their frequency in descending order
     * before being added to the list.
     *
     * @param ngram_list A reference to a vector of TokenSeq_t where the
     * n-grams' token sequences will be stored. It SHOULD BE EMPTY.
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
        PairMap_t pair_map;

        std::cout << "count token and pairs..." << std::endl;
        size_t total_num_tokens = 0;
        {
            ThreadPool thread_pool(this->config.num_workers);
            thread_pool.init();
            // lock striping
            std::vector<TokenDict_t> token_dict_vec(this->config.num_workers);
            std::vector<PairMap_t> pair_map_vec(this->config.num_workers);
            std::vector<std::mutex> locks(this->config.num_workers);

            std::vector<std::future<void>> futures;
            // for (const auto &token_seq : token_seq_vec) {
            for (size_t idx = 0; idx < token_seq_vec.size(); idx++) {
                total_num_tokens += token_seq_vec[idx].size();
                futures.emplace_back(thread_pool.submit([&, this, idx]() {
                    this->_count_token_and_pairs2(
                        token_seq_vec[idx],
                        token_dict_vec[idx % this->config.num_workers],
                        pair_map_vec[idx % this->config.num_workers],
                        locks[idx % this->config.num_workers]);
                }));
            }
            // Wait for all threads to finish
            for (auto &f : futures) {
                f.get();
            }

            // merge all dicts
            for (const auto &token_dict_tmp : token_dict_vec) {
                for (const auto &pair : token_dict_tmp) {
                    this->token_dict[pair.first] += pair.second;
                }
            }

            for (const auto &pair_map_tmp : pair_map_vec) {
                for (const auto &pair : pair_map_tmp) {
                    pair_map[pair.first] += pair.second;
                }
            }
        }

        // filter the mappings based on the minimum count
        filter_dict_by_freq(this->token_dict, this->config.min_token_count);
        filter_dict_by_freq(pair_map, this->config.min_token_count);

        // filter the pairs based on pmi
        std::cout << "filter pairs.." << std::endl;
        this->_filter_pairs(pair_map, total_num_tokens);

        // iterate over the token sequences again.
        // enumerate over ngram candidates
        std::cout << "iterate over token seq" << std::endl;
        for (auto &tok_seq : token_seq_vec) {
            TokenSeq_t ngram_candidate = {tok_seq[0]};
            for (size_t i = 1; i < tok_seq.size(); i++) {
                TokenPair_t tok_pair = {tok_seq[i - 1], tok_seq[i]};
                if (pair_map.find(tok_pair) != pair_map.end() &&
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

        // remove pairs to save memory
        {
            PairMap_t tmp;
            std::swap(pair_map, tmp);
        }

        filter_dict_by_freq(this->ngrams, this->config.min_ngram_freq);

        // renew ngram by freq
        std::cout << "renew ngram by freq..." << std::endl;
        {
            ThreadPool thread_pool(this->config.num_workers);
            thread_pool.init();
            NgramMap_t new_ngram_dict;
            std::vector<std::future<void>> futures;
            for (auto &tok_seq : token_seq_vec) {
                futures.emplace_back(
                    // everything except `this` is pass-by-reference
                    thread_pool.submit([&, this]() {
                        this->_get_new_ngram_by_freq(tok_seq, this->ngrams,
                                                     new_ngram_dict);
                    }));
            }
            // Wait for all threads to finish
            for (auto &f : futures) {
                f.get();
            }
            std::swap(this->ngrams, new_ngram_dict);
        }

        // this->ngrams = new_ngram_dict;
        std::cout << "filer dict by freq..." << std::endl;
        filter_dict_by_freq(this->ngrams, this->config.min_ngram_freq);
    }

    void find_ngrams_from_file(std::string fname) {
        std::ifstream file(fname);

        if (!file.is_open()) {
            std::cerr << "Cannot open the file." << std::endl;
            return;
        }

        std::cout << "count token and pairs(mem-eficient)..." << std::endl;
        PairMap_t pair_map;
        size_t total_num_tokens = 0;
        {
            ThreadPool thread_pool(this->config.num_workers);
            thread_pool.init();

            std::vector<std::future<void>> futures;
            std::string line;
            while (std::getline(file, line)) {
                TokenSeq_t token_seq;
                make_token_seq(line, token_seq);
                total_num_tokens += token_seq.size();
                futures.emplace_back(
                    // here, token_seq is copy-by-value instead of
                    // copy-by-reference
                    thread_pool.submit([&, this, token_seq]() {
                        this->_count_token_and_pairs2(
                            token_seq, this->token_dict, pair_map, this->mutex);
                    }));
            }
            // Wait for all threads to finish
            for (auto &f : futures) {
                f.get();
            }
        }

        // filter the mappings based on the minimum count
        filter_dict_by_freq(this->token_dict, this->config.min_token_count);
        filter_dict_by_freq(pair_map, this->config.min_token_count);

        // filter the pairs based on pmi
        std::cout << "filter pairs.." << std::endl;
        this->_filter_pairs(pair_map, total_num_tokens);

        // iterate over the token sequences again.
        // enumerate over ngram candidates
        std::cout << "iterate over token seq" << std::endl;
        std::string line;
        while (std::getline(file, line)) {
            TokenSeq_t tok_seq;
            make_token_seq(line, tok_seq);
            TokenSeq_t ngram_candidate = {tok_seq[0]};
            for (size_t i = 1; i < tok_seq.size(); i++) {
                TokenPair_t tok_pair = {tok_seq[i - 1], tok_seq[i]};
                if (pair_map.find(tok_pair) != pair_map.end() &&
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

        // remove pairs to save memory
        {
            PairMap_t tmp;
            std::swap(pair_map, tmp);
        }

        filter_dict_by_freq(this->ngrams, this->config.min_ngram_freq);

        // renew ngram by freq
        std::cout << "renew ngram by freq..." << std::endl;
        NgramMap_t new_ngram_dict;
        while (std::getline(file, line)) {
            TokenSeq_t tok_seq;
            make_token_seq(line, tok_seq);
            this->_get_new_ngram_by_freq(tok_seq, this->ngrams, new_ngram_dict);
        }

        this->ngrams = new_ngram_dict;
        std::cout << "filer dict by freq..." << std::endl;
        filter_dict_by_freq(this->ngrams, this->config.min_ngram_freq);
    }
};