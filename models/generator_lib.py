# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local helper for accessing generator.h5."""

import os

PATH_TO_GENERATOR_SPLIT_1 = os.path.join(
    os.sep,
    *os.path.split(__file__)[0].split('/'),
    "generator-split1.h5")

PATH_TO_GENERATOR_SPLIT_2 = os.path.join(
    os.sep,
    *os.path.split(__file__)[0].split('/'),
    "generator-split2.h5")
