/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MOSS-TTS-Nano end-to-end smoke driver. Loads a TRT engine bundle, runs
 * generate() with dummy zero token IDs, writes a 48kHz stereo float WAV.
 *
 * usage: moss_tts_nano_smoke <engine_dir> <output.wav>
 */

#include "runtime/mossTtsNanoRuntime.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

bool writeWavStereoF32(std::string const& path, std::vector<float> const& pcm)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    auto w4 = [&](char const* s) { out.write(s, 4); };
    auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<char const*>(&v), sizeof(v)); };
    auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<char const*>(&v), sizeof(v)); };
    uint32_t const dataBytes = static_cast<uint32_t>(pcm.size() * sizeof(float));
    w4("RIFF");
    u32(36u + dataBytes);
    w4("WAVE");
    w4("fmt ");
    u32(16);
    u16(3);                       // IEEE float PCM
    u16(2);                       // stereo
    u32(48000);                   // sample rate
    u32(48000u * 2u * 4u);        // byte rate
    u16(static_cast<uint16_t>(2 * 4)); // block align
    u16(32);                      // bits per sample
    w4("data");
    u32(dataBytes);
    out.write(reinterpret_cast<char const*>(pcm.data()), dataBytes);
    return static_cast<bool>(out);
}

int64_t elapsedMs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

} // namespace

int main(int argc, char** argv)
{
    using trt_edgellm::rt::MossTtsNanoRuntime;
    try
    {
        if (argc != 3)
        {
            std::fprintf(stderr, "usage: %s <engine_dir> <output.wav>\n", argv[0]);
            return 1;
        }
        std::string const engineDir = argv[1];
        std::string const outputWav = argv[2];

        std::cout << "MOSS smoke: loading engines from " << engineDir << "\n";
        MossTtsNanoRuntime runtime(engineDir, /*maxSlots=*/1, /*maxSeqLen=*/512);
        std::cout << "MOSS smoke: runtime ready, codec sr=" << runtime.codecSampleRate()
                  << " ch=" << runtime.codecChannels() << " ds=" << runtime.codecDownsampleRate()
                  << " n_vq=" << runtime.numVq() << "\n";

        auto guard = runtime.beginRequest();
        auto& slot = guard.slot();

        // Dummy inputs: 1 prefill token × row_width 17, all-zero.
        // Engines built without explicit shape profiles bake in the export-time
        // sample shape, so prefill is currently locked to seqLen=1. To accept
        // longer prompts, rebuild engines with --minShapes/--optShapes/--maxShapes.
        constexpr int32_t kSeqLen = 1;
        constexpr int32_t kRowWidth = 17;
        std::vector<int32_t> inputIds(static_cast<size_t>(kSeqLen) * kRowWidth, 0);
        std::vector<int32_t> attentionMask(static_cast<size_t>(kSeqLen), 1);
        std::vector<float> pcmOut;
        pcmOut.reserve(48000 * 2 * 10); // pre-reserve ~10s worth

        int64_t ttfaMs = -1;
        auto const start = std::chrono::steady_clock::now();

        MossTtsNanoRuntime::GenerateConfig cfg;
        cfg.maxNewFrames = 100;
        cfg.codecChunkFrames = 4;
        cfg.onFirstChunk = [&]() { ttfaMs = elapsedMs(start, std::chrono::steady_clock::now()); };

        runtime.generate(slot, inputIds, attentionMask, cfg, pcmOut);

        auto const end = std::chrono::steady_clock::now();

        if (!writeWavStereoF32(outputWav, pcmOut))
        {
            std::cerr << "SMOKE_FAIL: could not write " << outputWav << "\n";
            return 1;
        }

        size_t const stereoFrames = pcmOut.size() / 2;
        double const seconds = static_cast<double>(stereoFrames) / 48000.0;
        std::cout << "SMOKE_OK pcm_samples=" << pcmOut.size() << " stereo_frames=" << stereoFrames
                  << " seconds=" << seconds << " wall_ms=" << elapsedMs(start, end)
                  << " ttfa_ms=" << ttfaMs << "\n";
        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "SMOKE_FAIL " << e.what() << "\n";
        return 1;
    }
}
