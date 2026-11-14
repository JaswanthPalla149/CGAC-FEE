#pragma once

#include <cstdlib>
#include <iostream>

namespace test {

inline void check(bool condition, const char* expression, const char* file,
                  int line) {
    if (!condition) {
        std::cerr << file << ':' << line << ": check failed: " << expression
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace test

#define CHECK(...) \
    ::test::check(static_cast<bool>((__VA_ARGS__)), #__VA_ARGS__, __FILE__, __LINE__)
