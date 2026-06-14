// io.hpp — minimal binary readers for the Python-exported reference fixtures.
#pragma once
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

namespace pio {

inline FILE* open_rd(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + p);
    return f;
}
inline int32_t rd_i32(FILE* f) { int32_t v; if (std::fread(&v,4,1,f)!=1) throw std::runtime_error("rd i32"); return v; }
inline std::vector<double> rd_doubles(FILE* f, int n) {
    std::vector<double> v(n);
    if ((int)std::fread(v.data(), sizeof(double), n, f) != n) throw std::runtime_error("rd doubles");
    return v;
}
// File of the form: [int32 count][count float64]
inline std::vector<double> load_flat(const std::string& p) {
    FILE* f = open_rd(p);
    int32_t n = rd_i32(f);
    auto v = rd_doubles(f, n);
    std::fclose(f);
    return v;
}

} // namespace pio
