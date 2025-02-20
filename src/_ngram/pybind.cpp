#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FreqNgramFinder.hpp"
#include "PmiNgramFinder.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ngram, m) {
    py::class_<PmiNgramFinderConfig>(m, "PmiNgramFinderConfig")
        .def(py::init())
        .def_readwrite("min_ngram_freq", &PmiNgramFinderConfig::min_ngram_freq)
        .def_readwrite("min_ngram_len", &PmiNgramFinderConfig::min_ngram_len)
        .def_readwrite("max_ngram_len", &PmiNgramFinderConfig::max_ngram_len)
        .def_readwrite("min_pmi", &PmiNgramFinderConfig::min_pmi)
        .def_readwrite("min_token_count",
                       &PmiNgramFinderConfig::min_token_count)
        .def_readwrite("num_workers", &PmiNgramFinderConfig::num_workers);

    py::class_<PmiNgramFinder>(m, "PmiNgramFinder")
        .def(py::init<struct PmiNgramFinderConfig>())
        .def("find_ngrams_batched", &PmiNgramFinder::find_ngrams_batched)
        .def("find_ngrams_from_file", &PmiNgramFinder::find_ngrams_from_file)
        .def_property_readonly("token_freq", &PmiNgramFinder::get_tokens)
        .def("get_ngram_list", &PmiNgramFinder::ngram_to_list);

    py::class_<FreqNgramFinderConfig>(m, "FreqNgramFinderConfig")
        .def(py::init())
        .def_readwrite("min_freq", &FreqNgramFinderConfig::min_freq)
        .def_readwrite("min_ngram_len", &FreqNgramFinderConfig::min_ngram_len)
        .def_readwrite("max_ngram_len", &FreqNgramFinderConfig::max_ngram_len)
        .def_readwrite("secondary_filter",
                       &FreqNgramFinderConfig::secondary_filter)
        .def_readwrite("num_workers", &FreqNgramFinderConfig::num_workers);

    py::class_<FreqNgramFinder>(m, "FreqNgramFinder")
        .def(py::init<struct FreqNgramFinderConfig>())
        .def("find_ngrams_batched", &FreqNgramFinder::find_ngrams_batched)
        .def("get_ngram_list", &FreqNgramFinder::ngram_to_list);
}