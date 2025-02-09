#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "NgramFinder.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ngram, m) {
    py::class_<NgramFinderConfig>(m, "NgramFinderConfig")
        .def(py::init())
        .def_readwrite("min_ngram_freq", &NgramFinderConfig::min_ngram_freq)
        .def_readwrite("min_ngram_len", &NgramFinderConfig::min_ngram_len)
        .def_readwrite("max_ngram_len", &NgramFinderConfig::max_ngram_len)
        .def_readwrite("min_pmi", &NgramFinderConfig::min_pmi)
        .def_readwrite("min_token_count", &NgramFinderConfig::min_pmi)
        .def_readwrite("num_workers", &NgramFinderConfig::num_workers);

    py::class_<DnaNgramFinder>(m, "DnaNgramFinder")
        .def(py::init<struct NgramFinderConfig>())
        .def("find_ngrams_batched", &DnaNgramFinder::find_ngrams_batched)
        .def("find_ngrams_from_file", &DnaNgramFinder::find_ngrams_from_file)
        .def_property_readonly("token_freq", &DnaNgramFinder::get_tokens)
        .def("get_ngram_list", &DnaNgramFinder::ngram_to_list);
}