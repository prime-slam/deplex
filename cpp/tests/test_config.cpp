/**
 * Copyright (c) 2022, Arthur Saliou, Anastasiia Kornilova
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
#include <gtest/gtest.h>

#include <deplex/config.h>

#include "globals.hpp"

namespace deplex {
namespace {
TEST(ConfigInit, InvalidPath) { ASSERT_THROW(config::Config("__INVALID_PATH"), std::runtime_error); }

TEST(ConfigInit, MissingParameters) {
  auto config = config::Config(test_globals::invalid::missing_params_config);
  ASSERT_EQ(config.depth_sigma_coeff, config::Config().depth_sigma_coeff);
}
}  // namespace
}  // namespace deplex