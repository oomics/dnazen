#include "common.hpp"

void _count_token_and_pairs(const TokenSeq_t &token_seq,
                            TokenDict_t &token_dict, PairMap_t &pair_map) {
    if (token_seq.empty()) {
        // dummy variables so that they could be iterated.
        token_dict[0] = 0;
        pair_map[{0, 0}] = 0;
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

struct PmiNgramFinderConfig {
    size_t min_ngram_freq;
    size_t min_ngram_len;
    size_t max_ngram_len;
    double min_pmi;

    size_t min_token_count;

    int num_workers = 16;
};

class PmiNgramFinder {
   private:
    struct PmiNgramFinderConfig config;
    TokenDict_t token_dict;
    NgramMap_t ngrams;
    std::mutex mutex;

    /**
     * @brief Filter the pairs using pmi.
     */
    void _filter_pairs(PairMap_t &pairs, size_t total_num_pairs) {
        std::vector<TokenPair_t> keys_to_erase;
        for (auto it = pairs.begin(); it != pairs.end(); it++) {
            auto pair = it->first;
            const size_t count = it->second;

            const Token_t a = std::get<0>(pair);
            const Token_t b = std::get<1>(pair);

            if (this->token_dict.find(a) == this->token_dict.end() ||
                this->token_dict.find(b) == this->token_dict.end()) {
                keys_to_erase.push_back(pair);
            }

            double mi =
                (log2(total_num_pairs) + log2(count) -
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
                                const NgramMap_t &ngram_dict,
                                NgramMap_t &new_ngram_dict) {
        NgramMap_t local_ngram_dict;

        for (size_t idx = 0; idx < token_seq.size(); idx++) {
            for (size_t ngram_len = 1;
                 ngram_len <= this->config.max_ngram_len &&
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

        std::lock_guard<std::mutex> lock(mutex);
        for (const auto &pair : local_token_dict) {
            token_dict[pair.first] += pair.second;
        }
        for (const auto &pair : local_pair_map) {
            pairs[pair.first] += pair.second;
        }
    }

   public:
    PmiNgramFinder(PmiNgramFinderConfig config) : token_dict(), ngrams() {
        DETECT_KEY_INTERRUPT()
        this->config = config;
        if (this->config.min_ngram_len == 0) {
            std::cout << "Warning: Cannot have min_ngram_len set to 0. "
                         "Resetting to 1."
                      << std::endl;
            this->config.min_ngram_len = 1;
        }
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
        ThreadPool thread_pool(this->config.num_workers);
        thread_pool.init();

        PairMap_t pair_map;

        std::cout << "count token and pairs..." << std::endl;
        size_t total_num_pairs = 0;
        {
            std::vector<std::future<void>> futures;
            // lock striping
            std::vector<TokenDict_t> token_dict_vec(this->config.num_workers);
            std::vector<PairMap_t> pair_map_vec(this->config.num_workers);
            std::vector<std::mutex> locks(this->config.num_workers);

            for (size_t idx = 0; idx < token_seq_vec.size(); idx++) {
                total_num_pairs += token_seq_vec[idx].size();
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
            std::cout << "merging token and pairs frequency results..."
                      << std::endl;
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
        this->_filter_pairs(pair_map, total_num_pairs);

        // iterate over the token sequences again.
        // enumerate over ngram candidates
        std::cout << "iterate over token seq" << std::endl;
        for (auto &tok_seq : token_seq_vec) {
            if (tok_seq.size() < this->config.min_ngram_len) continue;
            TokenSeq_t ngram_candidate = {tok_seq[0]};
            for (size_t i = 1; i < tok_seq.size(); i++) {
                TokenPair_t tok_pair = {tok_seq[i - 1], tok_seq[i]};
                if (pair_map.find(tok_pair) != pair_map.end() &&
                    ngram_candidate.size() <= this->config.max_ngram_len) {
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
            std::vector<std::future<void>> futures;
            NgramMap_t new_ngram_dict;
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

        filter_dict_by_min_length(this->ngrams, this->config.min_ngram_len);
    }

    void find_ngrams_from_file(std::string fname) {
        std::ifstream file(fname);

        if (!file.is_open()) {
            std::cerr << "Cannot open the file." << std::endl;
            return;
        }

        std::cout << "count token and pairs(mem-eficient)..." << std::endl;
        PairMap_t pair_map;
        size_t total_num_pairs = 0;
        {
            ThreadPool thread_pool(this->config.num_workers);
            thread_pool.init();

            std::vector<std::future<void>> futures;
            std::string line;
            while (std::getline(file, line)) {
                TokenSeq_t token_seq;
                make_token_seq(line, token_seq);
                total_num_pairs += token_seq.size();
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
        this->_filter_pairs(pair_map, total_num_pairs);

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
                    ngram_candidate.size() <= this->config.max_ngram_len) {
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