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
"""Tabular data simulator."""

from pathlib import Path
from io import IOBase
from collections import OrderedDict

import numpy as np

# float epsilon
_EPS = 1e-9

# max. no. of bins in a column group
_COL_GRP_MAX_BIN = 10000

# min. normalized mutual information in a column group
_COL_GRP_MIN_NMI = 0.1

# max. no. of distinct values in a discrete column
_MAX_DIS_VAL = 256

# column type to data type map
# numeric - int, float
# discrete - cat, str
_DTYPE_MAP = {
    'int': int,
    'float': float,
    'cat': int,
    'str': str
}


class TabWriter:
    """Base class of simulated data writer."""

    def begin(self, schema, batch_size, num_rows):
        """
        Begins the writing.

        Args:
            schema (list[tuple[str, type]]): List of column name and data type tuples, possible data types are: `int`,
                `float` or `str`.
            batch_size (int): Maximum number of rows to be written in each write() call.
            num_rows (int): Total number of rows to be written.

        Raises:
            TypeError: Be raised for any argument or input type problem.
            ValueError: Be raised for any input value problem.
            IOError: Be raised for any I/O problem.
        """
        raise NotImplementedError

    def write(self, batch):
        """
        Writes a batch of column data.

        Args:
            batch (list[np.ndarray]): List of column data to be written, the length of data is inclusively between 1
                and batch_size.

        Raises:
            TypeError: Be raised for any argument or input type problem.
            ValueError: Be raised for any input value problem.
            IOError: Be raised for any I/O problem.
        """
        raise NotImplementedError

    def end(self):
        """
        Ends the writing.

        Raises:
            IOError: Be raised for any I/O problem.
        """
        raise NotImplementedError


class CsvTabWriter(TabWriter):
    """
    CSV writer of simulated data.

    Args:
        file (str, Path, IOBase): The file path or stream to be written to. If a str or Path is provided, then the file
            will be opened and closed automatically. If an IOBase is provider, caller has to close the stream on it's
            own.
    """
    def __init__(self, file):
        if not isinstance(file, (str, Path, IOBase)):
            raise TypeError(f'Argument "file" must be in type of str, Path or IOBase.')
        self._file = file
        self._fp = None
        self._close_on_end = False

    def begin(self, schema, batch_size, num_rows):
        """
        Begins the writing.

        Args:
            schema (list[tuple[str, type]]): List of column name and data type tuples, possible data types are: `int`,
                `float` or `str`.
            batch_size (int): Maximum number of rows to be written in each write() call.
            num_rows (int): Total number of rows to be written.

        Raises:
            TypeError: Be raised for any argument or input type problem.
            ValueError: Be raised for any input value problem.
            IOError: Be raised for any I/O problem.
        """
        del batch_size
        del num_rows
        if isinstance(self._file, (str, Path)):
            if isinstance(self._file, Path):
                self._fp = self._file.open(mode='w')
            else:
                self._fp = open(self._file, mode='w')
                self._close_on_end = True
        else:
            self._fp = self._file
            self._close_on_end = False

        header = ','.join([t[0] for t in schema])
        self._fp.write(header + '\n')

    def write(self, batch):
        """
        Writes a batch of column data.

        Args:
            batch (list[np.ndarray]): List of column data to be written, the length of data is inclusively between 1
                and batch_size.

        Raises:
            TypeError: Be raised for any argument or input type problem.
            ValueError: Be raised for any input value problem.
            IOError: Be raised for any I/O problem.
        """
        row_count = batch[0].size
        for r in range(row_count):
            row = ','.join([str(col[r]) for col in batch])
            self._fp.write(row + '\n')

    def end(self):
        """
        Ends the writing.

        Note:
            The file will be closed if it is provided as a str or Path object in the constructor.

        Raises:
            IOError: Be raised for any I/O problem.
        """
        if self._close_on_end:
            self._fp.close()
        self._fp = None


class ColDigest:
    """Column digest, data traits of a column."""
    def __init__(self):
        self.name = None
        self.idx = None
        self.type = None
        self.dtype = None
        self.is_numeric = None
        self.is_label = False
        self.bin_vals = None
        self.bin_count = None
        self.bin_width = None
        self.max_val = None
        self.min_val = None


class TabDigest:
    """Table digest, data traits of a table."""
    def __init__(self):
        self.columns = None
        self.col_groups = None
        self.col_group_dists = None
        self.label_col_idx = -1

    def save(self, file):
        """
        Saves to file.

        Args:
            file (str, Path, IOBase): The file path or stream to be written to. If a str or Path is provided, then the
                file will be opened and closed automatically. If an IOBase is provider, caller has to close the stream
                on it's own.

        Raises:
            TypeError: Be raised for any argument or input type problem.
            ValueError: Be raised for any input value problem.
            IOError: Be raised for any I/O problem.
        """
        raise NotImplementedError

    @staticmethod
    def load(file):
        """
        Loads from file.

        Args:
            file (str, Path, IOBase): The file path or stream to be read. If a str or Path is provided, then the
                file will be opened and closed automatically. If an IOBase is provider, caller has to close the stream
                on it's own.

        Returns:
            TabDigest - The loaded digest.

        Raises:
            TypeError: Be raised for any argument or input type problem.
            ValueError: Be raised for any input value problem.
            IOError: Be raised for any I/O problem.
        """
        raise NotImplementedError
