#pragma once

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
inline bool is_integer(const std::string &s) {
    if (s.empty() || (!isdigit(s[0]))) return false;

    char *p;
    strtol(s.c_str(), &p, 10);

    return (*p == 0);
}

/**
 * @brief parse the string of token sequence into vector of tokens
 *
 * the string of token seq is token separated by colons:
 * "<token1>:<token2>:...:<token5>:..."
 */
void make_token_seq(std::string &line, TokenSeq_t &token_seq) {
    size_t start = 0;
    size_t end = line.find(':');
    while (end != std::string::npos) {
        std::string token = line.substr(start, end - start);
        if (!is_integer(token)) {
            std::cerr << "Token " << token << " cannot be convertted to int."
                      << std::endl;
        }
        token_seq.push_back(std::stoul(token));

        start = end + 1;
        end = line.find(':', start);
    }

    // process the last string
    std::string token = line.substr(start);
    if (!is_integer(token)) {
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
    // size_t count = 0;
    for (auto it = dict.begin(); it != dict.end(); it++) {
        // std::cout<<"[debug] count="<<count<<std::endl;
        if (it->second < min_freq) {
            keys_to_erase.push_back(it->first);
        }
        // count++;
    }
    for (auto &key : keys_to_erase) {
        dict.erase(key);
    }
}

template <typename T>
void filter_dict_by_max_length(T &dict, size_t max_len) {
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
        auto key = it->first;
        if (key.size() > max_len) {
            keys_to_erase.push_back(it->first);
        }
    }
    for (auto &key : keys_to_erase) {
        dict.erase(key);
    }
}

template <typename T>
void filter_dict_by_min_length(T &dict, size_t min_len) {
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
        auto key = it->first;
        if (key.size() < min_len) {
            keys_to_erase.push_back(it->first);
        }
    }
    for (auto &key : keys_to_erase) {
        dict.erase(key);
    }
}

// a simple implementation of concurrent hashmap
// not optimized for performance but is enough to get the job done.
template <typename K, typename V, typename H = std::hash<K>>
class ConcurrentHashMap {
    typedef std::unordered_map<K, V, H> HashBucket_t;

    std::vector<HashBucket_t> buckets;
    std::vector<std::mutex> mutexes;
    H hash_function;

   private:
    size_t _get_bucket_index(const K &key) {
        size_t hash_value = hash_function(key);
        return hash_value % mutexes.size();
    }

    void _dump_bucket_to_unordered_map(HashBucket_t &target, size_t idx) {
        auto &bucket = this->buckets[idx];

        if (bucket.empty()) return;

        std::unique_lock lock(this->mutexes[idx]);
        for (auto &pair : bucket) {
            target[pair.first] += pair.second;
        }
    }

   public:
    ConcurrentHashMap(size_t num_buckets = 16)
        : buckets(num_buckets), mutexes(num_buckets) {}

    void insert(const K &key, const V &value) {
        auto index = _get_bucket_index(key);
        auto bucket = this->buckets[index];

        std::lock_guard<std::mutex> lock(this->mutexes[index]);
        bucket.insert(key, value);
    }

    /**
     * @brief Index a value by key (only supports lvalue and rvalue keys).
     *
     * Thread-safely is only guaranteed for retrieving the value.
     * DO NOT try to update the value using index (e.g. y[idx] += c) in
     * multi-threading.
     */
    template <typename KeyType>
    V &operator[](KeyType &&key) {
        static_assert(std::is_convertible_v<KeyType, K>,
                      "KeyType must be convertible to K");

        auto index = _get_bucket_index(key);
        auto &bucket = this->buckets[index];

        std::lock_guard<std::mutex> lock(this->mutexes[index]);
        return bucket[key];
    }

    void update_key_addition(const K &key, const V val) {
        auto index = _get_bucket_index(key);
        auto &bucket = this->buckets[index];

        std::lock_guard<std::mutex> lock(this->mutexes[index]);
        bucket[key] += val;
    }

    void update_key_addition(K &&key, const V val) {
        auto index = _get_bucket_index(key);
        auto &bucket = this->buckets[index];

        std::lock_guard<std::mutex> lock(this->mutexes[index]);
        bucket[key] += val;
    }

    /**
     * @brief Apply the function to all buckets.
     */
    void parallel_apply(std::function<void(HashBucket_t &)> &&func) {
        ThreadPool thread_pool(this->buckets.size());
        thread_pool.init();
        std::vector<std::future<void>> futures;
        futures.reserve(this->buckets.size());
        for (size_t idx = 0; idx < this->buckets.size(); idx++) {
            futures.emplace_back(thread_pool.submit(
                [&, this, idx]() { func(this->buckets[idx]); }));
        }

        // Wait for all threads to finish
        for (auto &f : futures) {
            f.get();
        }
    }

    void to_unordered_map(HashBucket_t &target) {
        for (size_t idx = 0; idx < buckets.size(); idx++) {
            this->_dump_bucket_to_unordered_map(target, idx);
        }
    }
};