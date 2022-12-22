/**
 * Copyright 2022 prime-slam
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <benchmark/benchmark.h>
#include <deplex/config.h>
#include <deplex/plane_extractor.h>
#include <deplex/utils/eigen_io.h>

#include "globals.hpp"

namespace deplex {
namespace {
void BM_SINGLE_TUM(benchmark::State& state) {
  auto config = config::Config(bench_globals::tum::config);
  auto algorithm = PlaneExtractor(480, 640, config);

  auto cloud = utils::readPointCloudCSV(bench_globals::tum::sample_image_points);
  for (auto _ : state) {
    for (int i = 0; i < 60; ++i) {
      algorithm.process(cloud);
    }
  }
}

BENCHMARK(BM_SINGLE_TUM)->Unit(benchmark::TimeUnit::kSecond)->Iterations(30);
}  // namespace
}  // namespace deplex

BENCHMARK_MAIN();