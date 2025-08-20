#pragma once

#include "common.h"

template <int N, typename value_t_>
struct Array {
    using value_t = value_t_;

    value_t _data[N];

    FA_DEVICE_CONSTEXPR value_t *data() { return _data; }
    FA_DEVICE_CONSTEXPR const value_t *data() const { return _data; }

    FA_DEVICE_CONSTEXPR void fill(value_t val) {
        FA_UNROLL
        for (int i = 0; i < N; ++i) {
            _data[i] = value_t(val);
        }
    }

    FA_DEVICE_CONSTEXPR void zero() { fill(0); }

    FA_DEVICE_CONSTEXPR value_t &operator[](size_t idx) { return _data[idx]; }
    FA_DEVICE_CONSTEXPR value_t operator[](size_t idx) const {
        return _data[idx];
    }

    FA_DEVICE_CONSTEXPR static size_t size() { return N; }

    template <typename Other>
    FA_DEVICE_CONSTEXPR void copy(const Other &other) {
        static_assert(std::is_same<value_t, typename Other::value_t>::value,
                      "Arrays must have the same value type");
        static_assert(N == Other::size(), "Arrays must have the same size");

        FA_UNROLL
        for (int i = 0; i < N; ++i) {
            _data[i] = other[i];
        }
    }
};

template <int N, typename value_t, int Alignment = 16>
struct __align__(Alignment) ArrayAligned : public Array<N, value_t> {};
