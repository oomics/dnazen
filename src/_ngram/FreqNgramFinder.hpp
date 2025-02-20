#include "common.hpp"

struct FreqNgramFinderConfig {
    size_t min_freq;
    size_t min_ngram_len;
    size_t max_ngram_len;

    bool secondary_filter = true;

    int num_workers = 16;
};

// Freq-ngram
class FreqNgramFinder {
    typedef ConcurrentHashMap<TokenSeq_t, uint64_t, VectorHash>
        ConcurrentNgramMap_t;

   private:
    struct FreqNgramFinderConfig config;
    NgramMap_t ngram_dict;

    void _count_ngram_freq(
        const TokenSeq_t &token_seq,
        ConcurrentNgramMap_t &target_ngram_dict  // the ngram dict to update
    ) {
        if (token_seq.empty()) {
            std::cout << "Warning: token_seq is empty" << std::endl;
            return;
        }

        NgramMap_t local_ngram_dict;
        for (size_t i = 0; i < token_seq.size(); i++) {
            for (size_t ngram_len = this->config.min_ngram_len;
                 ngram_len <= this->config.max_ngram_len &&
                 i + ngram_len <= token_seq.size();
                 ngram_len++) {
                TokenSeq_t ngram = TokenSeq_t(
                    token_seq.begin() + i, token_seq.begin() + i + ngram_len);
                local_ngram_dict[ngram]++;
            }
        }

        // update the target ngram
        for (const auto &pair : local_ngram_dict) {
            // target_ngram_dict[pair.first] += pair.second;
            target_ngram_dict.update_key_addition(pair.first, pair.second);
        }
    }

    void _count_ngram_without_overlap(const TokenSeq_t &token_seq,
                                      const NgramMap_t &ngram_dict,
                                      ConcurrentNgramMap_t &target_ngram_dict) {
        if (token_seq.size() < this->config.min_ngram_len) return;
        NgramMap_t local_ngram_dict;

        TokenSeq_t ngram_candidate = TokenSeq_t(
            token_seq.begin(), token_seq.begin() + this->config.min_ngram_len);
        size_t i = this->config.min_ngram_len;
        while (i < token_seq.size()) {
            if (ngram_dict.find(ngram_candidate) != ngram_dict.end() &&
                ngram_candidate.size() < this->config.max_ngram_len) {
                // the current ngram exists
                ngram_candidate.push_back(token_seq[i]);
                i++;
            } else if (i + this->config.min_ngram_len > token_seq.size()) {
                break;
            } else if (ngram_candidate.size() < this->config.min_ngram_len) {
                ngram_candidate = TokenSeq_t(
                    token_seq.begin() + i,
                    token_seq.begin() + i + this->config.min_ngram_len);
                i += this->config.min_ngram_len;
            } else {
                local_ngram_dict[ngram_candidate]++;
                ngram_candidate = TokenSeq_t(
                    token_seq.begin() + i,
                    token_seq.begin() + i + this->config.min_ngram_len);
                i += this->config.min_ngram_len;
            }
        }

        // deal with the last one
        if (ngram_dict.find(ngram_candidate) != ngram_dict.end() &&
            ngram_candidate.size() < this->config.max_ngram_len) {
            local_ngram_dict[ngram_candidate]++;
        }

        // we don't want to waste our time
        for (const auto &pair : local_ngram_dict) {
            target_ngram_dict.update_key_addition(pair.first, pair.second);
        }
    }

   public:
    FreqNgramFinder(FreqNgramFinderConfig config) : ngram_dict() {
        DETECT_KEY_INTERRUPT()
        this->config = config;
        std::cout << "[debug] min_freq=" << config.min_freq << std::endl;
        if (this->config.min_ngram_len == 0) {
            std::cout << "Warning: Cannot have min_ngram_len set to 0. "
                         "Resetting to 1."
                      << std::endl;
            this->config.min_ngram_len = 1;
        }
    }

    NgramMap_t &get_ngrams() { return this->ngram_dict; }

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
            this->ngram_dict.begin(), this->ngram_dict.end());

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
        std::cout << "count ngrams..." << std::endl;
        {
            std::vector<std::future<void>> futures;
            futures.reserve(token_seq_vec.size());
            size_t num_locks = 2 * this->config.num_workers;
            ConcurrentNgramMap_t concurr_ngram_map(num_locks);

            for (size_t idx = 0; idx < token_seq_vec.size(); idx++) {
                futures.emplace_back(thread_pool.submit([&, this, idx]() {
                    this->_count_ngram_freq(token_seq_vec[idx],
                                            concurr_ngram_map);
                }));
            }
            // Wait for all threads to finish
            for (auto &f : futures) {
                f.get();
            }

            // concurr_ngram_map.to_unordered_map(this->ngram_dict);
            std::cout << "Filtering ngram by frequency..." << std::endl;
            concurr_ngram_map.parallel_apply([this](NgramMap_t &ngram_dict) {
                filter_dict_by_freq(ngram_dict, this->config.min_freq);
            });
            std::cout << "merging frequency results..." << std::endl;
            concurr_ngram_map.to_unordered_map(
                this->ngram_dict);  // TODO: optimization. No need to convert
                                    // ConcurrentHashMap to unordered_map here.
                                    // Try it after we finish the second
                                    // iteration.
        }

        if (!this->config.secondary_filter) {
            return;
        }

        // re-count
        std::cout << "iterate over token seq to re-count frequency"
                  << std::endl;
        {
            std::vector<std::future<void>> futures;
            futures.reserve(token_seq_vec.size());
            size_t num_locks = 2 * this->config.num_workers;
            ConcurrentNgramMap_t concurr_ngram_map(num_locks);
            for (size_t idx = 0; idx < token_seq_vec.size(); idx++) {
                futures.emplace_back(thread_pool.submit([&, this, idx]() {
                    this->_count_ngram_without_overlap(token_seq_vec[idx],
                                                       this->ngram_dict,
                                                       concurr_ngram_map);
                }));
            }
            // Wait for all threads to finish
            for (auto &f : futures) {
                f.get();
            }

            std::cout << "Filtering ngram by frequency(2)..." << std::endl;
            concurr_ngram_map.parallel_apply([this](NgramMap_t &ngram_dict) {
                filter_dict_by_freq(ngram_dict, this->config.min_freq);
            });

            // merge all dicts
            std::cout << "merging frequency results(2)..." << std::endl;
            this->ngram_dict.clear();
            NgramMap_t tmp_ngram_dict;
            concurr_ngram_map.to_unordered_map(tmp_ngram_dict);
            std::swap(tmp_ngram_dict, this->ngram_dict);
        }
    }
};