#include "zen/kernels.h"
#include "zen/quant.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
void enable_q8_acts() {
#ifdef _WIN32
    _putenv_s("ZEN_CPU_Q8_ACT", "1");
#else
    setenv("ZEN_CPU_Q8_ACT", "1", 1);
#endif
}
}

int main() {
    constexpr std::size_t cols = 256;
    constexpr std::size_t rows = 129; // exercises the parallel row split
    std::vector<float> x(cols);
    for (std::size_t i = 0; i < cols; ++i)
        x[i] = static_cast<float>(static_cast<int>((i * 37 + 11) % 257) - 128) / 127.0f;
    // Include positive/negative half-way cases while keeping a block max of 1.
    for (std::size_t b = 0; b < cols / 32; ++b) {
        x[b * 32] = 1.0f;
        x[b * 32 + 1] = 0.5f / 127.0f;
        x[b * 32 + 2] = -1.5f / 127.0f;
    }

    std::vector<std::int8_t> q(cols);
    std::vector<float> scales(cols / 32);
    if (!zen::quantise_acts_q8_0_avx2(x.data(), cols, q.data(), scales.data())) {
        std::cerr << "activation quantisation rejected valid shape\n";
        return 1;
    }
    for (std::size_t b = 0; b < cols / 32; ++b) {
        float amax = 0.0f;
        for (std::size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        if (scales[b] != d) {
            std::cerr << "activation block scale mismatch at block " << b << '\n';
            return 1;
        }
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        for (std::size_t i = 0; i < 32; ++i) {
            float v = x[b * 32 + i] * inv;
            v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
            if (q[b * 32 + i] != static_cast<std::int8_t>(std::lround(v))) {
                std::cerr << "activation quant mismatch at element " << b * 32 + i << '\n';
                return 1;
            }
        }
    }

    // Build Q8_0 weights with an fp16 scale of 1.0 and deterministic signed bytes.
    constexpr std::size_t row_bytes = (cols / 32) * 34;
    std::vector<std::uint8_t> w(rows * row_bytes);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < cols / 32; ++b) {
            auto* block = w.data() + r * row_bytes + b * 34;
            block[0] = 0x00;
            block[1] = 0x3c;
            for (std::size_t i = 0; i < 32; ++i)
                block[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>((r * 13 + b * 7 + i * 3) % 255) - 127));
        }
    }

    std::vector<float> output(rows);
    enable_q8_acts();
    const auto* f = zen::quant::find("Q8_0");
    if (!f) {
        std::cerr << "Q8_0 format missing from registry\n";
        return 1;
    }
    if (!zen::quant::gemv(*f, w.data(), x.data(), output.data(), rows, cols, 1.0f, 4)) {
        std::cerr << "shared-activation Q8_0 GEMV dispatch failed (row_bytes="
                  << zen::quant::row_bytes(*f, cols) << ")\n";
        return 1;
    }
    for (std::size_t r = 0; r < rows; ++r) {
        double expected = 0.0;
        for (std::size_t b = 0; b < cols / 32; ++b) {
            const auto* block = w.data() + r * row_bytes + b * 34;
            std::int32_t dot = 0;
            for (std::size_t i = 0; i < 32; ++i)
                dot += static_cast<std::int8_t>(block[2 + i]) * static_cast<std::int32_t>(q[b * 32 + i]);
            expected += static_cast<double>(dot) * scales[b];
        }
        if (std::fabs(static_cast<double>(output[r]) - expected) > 2e-4 * std::max(1.0, std::fabs(expected))) {
            std::cerr << "shared-activation GEMV mismatch at row " << r << '\n';
            return 1;
        }
    }

    std::cout << "Q8_0 vector activation quantisation and shared-row GEMV pass\n";
    return 0;
}
