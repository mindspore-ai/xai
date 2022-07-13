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
import argparse

import mindspore_xai
from mindspore_xai.tool.tab.cli import cli_tabsim, cli_tabdig


def cli_entry():
    """Entry point for XAI CLI."""
    parser = argparse.ArgumentParser(
        prog='mindspore_xai',
        description='XAI CLI entry point (version: {})'.format(mindspore_xai.__version__),
        allow_abbrev=False)
    subparsers = parser.add_subparsers(help='sub-command help', dest='subparser_name')

    # tabular digest
    parser_tabdig = subparsers.add_parser('tabdig', help='digest tabular data', allow_abbrev=False)
    parser_tabdig.add_argument(type=str, dest='real_datafile',
                               help='path of the real CSV table to be simulated.')
    parser_tabdig.add_argument(type=str, dest='digest_file',
                               help='path of the digest file to be saved.')
    parser_tabdig.add_argument('--bins', type=int, dest='num_bins', required=False, choices=range(2, 33),
                               default=10, metavar="[2-32]",
                               help='[optional] number of bins (2-32) for discretizing numeric columns, default: 10')
    parser_tabdig.add_argument('--clip-sd', type=float, dest='clip_sd', required=False, default=3,
                               help='[optional] number of standard deviations away from the mean that defines the '
                                    'outliers, outlier values will be clipped. default: 3, setting to 0 or less will '
                                    'disable the value clipping.')

    # tabular simulate
    parser_tabsim = subparsers.add_parser('tabsim', help='simulate tabular data', allow_abbrev=False)
    parser_tabsim.add_argument(type=str, dest='digest_file',
                               help='path of the digest file of the real data.')
    parser_tabsim.add_argument(type=str, dest='sim_datafile',
                               help='path of the simulated CSV table.')
    parser_tabsim.add_argument(type=int, dest='rows',
                               help='number of rows to be generated to <sim datafile>')
    parser_tabsim.add_argument('--batch-size', type=int, dest='batch_size', required=False, default=10000,
                               help='[optional] number of rows in each batch, default: 10000')
    parser_tabsim.add_argument('--noise', type=float, dest='noise', required=False, default=0.0,
                               help='[optional] 0.0-1.0 noise level of value picking probabilities, 0.0 means flows '
                                    'exactly the digested joint distributions, higher the noise level more even the '
                                    'probabilities. default: 0.0')

    args = vars(parser.parse_args())
    if args['subparser_name'] == 'tabdig':
        cli_tabdig(**args)
    elif args['subparser_name'] == 'tabsim':
        cli_tabsim(**args)
    else:
        raise ValueError("usage: mindspore_xai [-h] {tabdig,tabsim} ...")


if __name__ == '__main__':
    cli_entry()
