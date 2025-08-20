#pragma once

namespace flash {

constexpr int constexpr_min(int a, int b) { return (a < b) ? a : b; }

constexpr int constexpr_log2_floor(int n) { return std::__bit_width(n) - 1; }

} // namespace flash
