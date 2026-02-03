#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <stdexcept>

#include "khmer-segmenter.h"

namespace py = pybind11;

class KhmerSegmenter {
public:
  KhmerSegmenter() {
    ctx_ = khmer_init();
    if (!ctx_) {
      throw std::runtime_error("Failed to initialize Khmer segmenter");
    }
  }

  ~KhmerSegmenter() {
    if (ctx_) {
      khmer_free(ctx_);
    }
  }

  // Disable copy
  KhmerSegmenter(const KhmerSegmenter&) = delete;
  KhmerSegmenter& operator=(const KhmerSegmenter&) = delete;

  // Enable move
  KhmerSegmenter(KhmerSegmenter&& other) noexcept : ctx_(other.ctx_) {
    other.ctx_ = nullptr;
  }

  KhmerSegmenter& operator=(KhmerSegmenter&& other) noexcept {
    if (this != &other) {
      if (ctx_) {
        khmer_free(ctx_);
      }
      ctx_ = other.ctx_;
      other.ctx_ = nullptr;
    }
    return *this;
  }

  std::string segment(const std::string& text) const {
    if (!ctx_) {
      throw std::runtime_error("Segmenter not initialized");
    }

    char* result = khmer_segment(ctx_, text.c_str());
    if (!result) {
      return "";
    }

    std::string output(result);
    khmer_free_string(result);
    return output;
  }

  std::vector<std::string> tokenize(const std::string& text) const {
    std::string segmented = segment(text);
    std::vector<std::string> tokens;

    if (segmented.empty()) {
      return tokens;
    }

    size_t start = 0;
    size_t end = segmented.find('|');

    while (end != std::string::npos) {
      tokens.push_back(segmented.substr(start, end - start));
      start = end + 1;
      end = segmented.find('|', start);
    }

    if (start < segmented.size()) {
      tokens.push_back(segmented.substr(start));
    }

    return tokens;
  }

private:
  struct khmer_context* ctx_;
};

// Global instance for simple API
static KhmerSegmenter* g_segmenter = nullptr;

static void ensure_initialized() {
  if (!g_segmenter) {
    g_segmenter = new KhmerSegmenter();
  }
}

static std::string segment(const std::string& text) {
  ensure_initialized();
  return g_segmenter->segment(text);
}

static std::vector<std::string> tokenize(const std::string& text) {
  ensure_initialized();
  return g_segmenter->tokenize(text);
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "Khmer Neural Segmenter - Fast word segmentation for Khmer text";

  // Class-based API
  py::class_<KhmerSegmenter>(m, "KhmerSegmenter")
    .def(py::init<>())
    .def("segment", &KhmerSegmenter::segment,
         py::arg("text"),
         "Segment Khmer text and return pipe-delimited words")
    .def("tokenize", &KhmerSegmenter::tokenize,
         py::arg("text"),
         "Segment Khmer text and return a list of words")
    .def("__call__", &KhmerSegmenter::tokenize,
         py::arg("text"),
         "Segment Khmer text and return a list of words");

  // Simple function API (uses global instance)
  m.def("segment", &segment,
        py::arg("text"),
        "Segment Khmer text and return pipe-delimited words");

  m.def("tokenize", &tokenize,
        py::arg("text"),
        "Segment Khmer text and return a list of words");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
