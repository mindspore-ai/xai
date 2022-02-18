# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Command module."""
import csv

from .tab_sim import CsvTabDigest, TabSim, CsvTabWriter


def cli_tabdig(subparser_name, real_datafile, digest_file, num_bins, clip_sd):
    """Entry point for XAI tabdig."""
    if subparser_name != "tabdig":
        raise ValueError('argument "subparser_name" must be "tabdig"')
    digest = CsvTabDigest(num_bins, clip_sd)
    with open(real_datafile, 'r') as f:
        digest.digest(csv.reader(f))
    digest.save(digest_file)


def cli_tabsim(subparser_name, digest_file, sim_datafile, rows, batch_size, noise):
    """Entry point for XAI tabsim."""
    if subparser_name != "tabsim":
        raise ValueError('argument "subparser_name" must be "tabsim"')
    digest = CsvTabDigest.load(digest_file)
    sim = TabSim(digest, batch_size=batch_size)
    with open(sim_datafile, 'w') as f:
        sim.generate(rows, CsvTabWriter(f), noise=noise)
