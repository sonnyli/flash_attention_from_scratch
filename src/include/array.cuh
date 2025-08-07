#pragma once

#include "common.h"

template <int N, typename value_t>
struct Array {
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

    FA_DEVICE_CONSTEXPR value_t &operator[](int idx) { return _data[idx]; }
    FA_DEVICE_CONSTEXPR value_t operator[](int idx) const { return _data[idx]; }

    FA_DEVICE_CONSTEXPR static int size() { return N; }
};

template <int N, typename value_t, int Alignment = 16>
struct __align__(Alignment) ArrayAligned : public Array<N, value_t> {};
