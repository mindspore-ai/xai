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


class CsvTabDigest(TabDigest):
    r"""
    CSV table digest, data traits of a CSV table.

    Note:
        The expected header format is: "<name>|<type>,<name>|<type>,<name>|<type>,...". <name> is column name, allowed
        patten is `[0-9a-zA-Z_\-]+`. <type> is column type, must be one of "int", "float", "cat" and "str". "int" and
        "float" are numeric columns while "cat" and "str" are discrete columns. The maximum number of distinct values
        of discrete columns is 256. The underlying data type of "cat" is integer without continuous assumption while
        "str" values' allowed patten is `[0-9a-zA-Z_\-\+\.]*`. Example header: "col_A|int,col_B|float,col_C|cat". User
        may optionally specify a label column by adding a "*" character before the column name, example:
        "col_A|int,col_B|float,*col_C|cat", only "cat" and "str" columns can be label. At most one label column is
        allowed.

    Args:
        num_bins (int): Number of bins for numeric columns. Default: 10.
        clip_sd (float, optional): Number of standard deviations for clipping numeric column values. Disable the
            clipping by providing None. Default: 3.

    Raises:
        TypeError: Be raised for any argument or input type problem.
        ValueError: Be raised for any input value problem.
    """
    def __init__(self, num_bins=10, clip_sd=3):
        super().__init__()
        self.num_bins = num_bins
        self.clip_sd = clip_sd
        self._values = None
        self._distincts = None
        self._bin_idxs = None
        self._low_nmi = None
        self._cache = None

    def digest(self, csv_reader, col_types=None, label_col=None):
        """
        Digest a CSV data file.

        Args:
            csv_reader (Iterable[Iterable[str]]): CSV reader of the data file.
            col_types (list[str], optional): Column types, the length must be same as the number of columns in the CSV.
                It overrides the column type information in the CSV file.
            label_col (str, optional): The label column name. It overrides the label specification in the CSV file.

        Raises:
            TypeError: Be raised for any argument or input type problem.
            ValueError: Be raised for any input value problem.
            IOError: Be raised for any I/O problem.
        """
        self._reset()
        self._cache = dict()
        header = True
        for row in csv_reader:
            if header:
                self._read_header(row, col_types, label_col)
                header = False
            else:
                self._read_record(row)

        self._proc_columns()
        self._calc_nmi_mat()
        self._group_columns()
        self._cleanup()

    def _reset(self):
        """Reset all internal states and caches."""
        self.columns = None
        self.col_groups = None
        self.col_group_dists = None
        self._cleanup()

    def _cleanup(self):
        """Cleanup internal states and caches."""
        self._values = None
        self._distincts = None
        self._bin_idxs = None
        self._low_nmi = None
        self._cache = None

    def _read_header(self, row, col_types, label_col):
        """Read the header row."""
        self.columns = [self._new_column(i, title, col_types, label_col) for i, title in enumerate(row)]
        for i, col in enumerate(self.columns):
            if col.is_label:
                if self.label_col_idx >= 0:
                    raise IOError('There are more than 1 label columns.')
                self.label_col_idx = i

        self._values = [[] for _ in row]
        self._distincts = [OrderedDict() for _ in row]
        self._bin_idxs = [None] * len(row)

    def _new_column(self, idx, title, col_types, label_col):
        """Create a new column digest."""
        splited = title.split('|')
        col_name = splited[0].strip()
        if not col_name or (len(col_name) == 1 and col_name[0] == '*'):
            raise IOError('Column name can not be empty.')
        if col_name[0] == '*':
            is_label = True
            col_name = col_name[1:]
        else:
            is_label = False
        if col_types:
            col_type = col_types[idx]
        else:
            if len(splited) != 2:
                raise IOError(f'Expecting a column tile of "<name>|<type>".')
            col_type = splited[1].strip()

        if label_col:
            is_label = (col_name == label_col)

        if is_label and col_type not in ('cat', 'str'):
            raise TypeError(f'Label column type must be either "cat" or "str".')

        try:
            dtype = _DTYPE_MAP[col_type]
        except KeyError:
            raise TypeError(f'Unrecognized column[{idx}] type:{col_type}.')
        col = ColDigest()
        col.name = col_name
        col.idx = idx
        col.type = col_type
        col.dtype = dtype
        col.is_numeric = col_type in ('int', 'float')
        col.is_label = is_label
        return col

    def _read_record(self, row):
        """Read a data row."""
        for i, val_str in enumerate(row):
            col = self.columns[i]
            val = col.dtype(val_str)
            if col.is_numeric:
                self._values[i].append(val)
                continue
            idx = self._distincts[i].get(val, -1)
            if idx == -1:
                idx = len(self._distincts[i])
                if idx >= _MAX_DIS_VAL:
                    raise IOError(f'Number of distinct values of column[{i}] is more than {_MAX_DIS_VAL}.')
                self._distincts[i][val] = idx
            self._values[i].append(idx)

    def _proc_columns(self):
        """Process all columns."""
        for col in self.columns:
            if col.is_numeric:
                self._proc_num_col(col)
            else:
                self._proc_disc_col(col)
        self._distincts = None
        self._values = None

    def _proc_disc_col(self, col):
        """Process discrete column."""
        col.bin_vals = np.array(list(self._distincts[col.idx].keys()), dtype=col.dtype)
        col.bin_count = col.bin_vals.shape[0]
        self._bin_idxs[col.idx] = np.array(self._values[col.idx], dtype=int)

    def _proc_num_col(self, col):
        """Process numeric column."""
        values = np.array(self._values[col.idx], dtype=col.dtype)
        if self.clip_sd is not None:
            sd = np.std(values)
            avg = np.mean(values)
            radius = self.clip_sd * sd
            clip_min = avg - radius
            clip_max = avg + radius
            if col.dtype is int:
                clip_min = int(np.floor(clip_min))
                clip_max = int(np.ceil(clip_max))
            values = np.clip(values, clip_min, clip_max)
        col.min_val = values.min()
        col.max_val = values.max()
        rng = col.max_val - col.min_val
        if col.dtype is int:
            rng += 1
        min_rng = self.num_bins if col.dtype is int else _EPS
        if rng < min_rng:
            if col.dtype is int:
                col.bin_width = 1
                col.bin_count = rng
            else:
                col.bin_width = 0
                col.bin_count = 1
        else:
            col.bin_width = rng / self.num_bins
            col.bin_count = self.num_bins
            if col.dtype is int:
                col.bin_width = int(np.round(col.bin_width))
                if col.min_val + col.bin_width * col.bin_count < col.max_val:
                    col.bin_count += 1
        bins = [col.min_val + col.bin_width * i for i in range(col.bin_count)]
        col.bin_vals = np.array(bins, dtype=col.dtype)
        bins.append(bins[-1] + col.bin_width)
        bin_idxs = np.digitize(values, bins) - 1
        bin_idxs = np.clip(bin_idxs, 0, None)
        self._bin_idxs[col.idx] = bin_idxs

    def _calc_nmi_mat(self):
        """Compute the lower normalized mutual information matrix."""
        col_count = len(self.columns)
        self._low_nmi = np.full((col_count, col_count), -1, dtype=float)
        for i in range(1, col_count):
            for j in range(i):
                self._low_nmi[i, j] = self._calc_nmi(i, j)

    def _calc_nmi(self, i, j):
        """Compute the normalized mutual information of 2 columns."""
        mi = 0.0
        hx = 0.0
        hy = 0.0
        calc_hy = True
        rec_count = self._bin_idxs[0].shape[0]
        for x in range(self.columns[i].bin_vals.size):
            mkey = f'mask_{i}.{x}'
            pkey = f'prob_{i}.{x}'
            mask_x = self._cache.get(mkey, None)
            if mask_x is None:
                mask_x = (self._bin_idxs[i] == x)
                x_prob = np.sum(mask_x) / rec_count
                self._cache[mkey] = mask_x
                self._cache[pkey] = x_prob
            else:
                x_prob = self._cache[pkey]

            hx -= x_prob * np.log(x_prob)
            for y in range(self.columns[j].bin_vals.size):
                mkey = f'mask_{j}.{y}'
                pkey = f'prob_{j}.{y}'
                mask_y = self._cache.get(mkey, None)
                if mask_y is None:
                    mask_y = (self._bin_idxs[j] == y)
                    y_prob = np.sum(mask_y) / rec_count
                    self._cache[mkey] = mask_y
                    self._cache[pkey] = y_prob
                else:
                    y_prob = self._cache[pkey]

                if calc_hy:
                    hy -= y_prob * np.log(y_prob)
                xy_count = np.sum(mask_x & mask_y)
                xy_prob = xy_count / rec_count
                mi += xy_prob * np.log((xy_prob + _EPS)/(x_prob * y_prob + _EPS))
            calc_hy = False

        if hx < _EPS or hy < _EPS:
            return 0.0

        return mi / np.sqrt(hx * hy)

    def _group_columns(self):
        """Group columns by mutual information."""
        self.col_groups = []
        self.col_group_dists = []

        low_nmi = self._low_nmi.copy()
        grouped = []

        if self.label_col_idx >= 0:
            group = [self.label_col_idx]
            self._add_col_group(group)
            grouped.extend(group)
            low_nmi[self.label_col_idx, :] = -1
            low_nmi[:, self.label_col_idx] = -1

        while True:
            group = self._find_col_group(low_nmi)
            if not group:
                break
            low_nmi[group] = -1
            self._add_col_group(group)
            grouped.extend(group)
        grouped = set(grouped)
        for i in range(len(self.columns)):
            if i not in grouped:
                self._add_col_group([i])

        self._low_nmi = None
        self._bin_idxs = None

    def _find_col_group(self, low_nmi):
        """Find the next column group by mutual information."""
        max_ij = np.unravel_index(np.argmax(low_nmi, axis=None), low_nmi.shape)
        if low_nmi[max_ij] < _COL_GRP_MIN_NMI:
            return None
        bins = self.columns[max_ij[0]].bin_vals.size * self.columns[max_ij[1]].bin_vals.size
        if bins > _COL_GRP_MAX_BIN:
            raise RuntimeError(f'_COL_GRP_MAX_BIN:{_COL_GRP_MAX_BIN} is too small.')
        group = list(max_ij)
        # prevent group members be selected in the following loop
        low_nmi[:, group] = -1

        while bins < _COL_GRP_MAX_BIN:
            max_ij = np.unravel_index(np.argmax(low_nmi[group, :], axis=None), (len(group), low_nmi.shape[1]))
            max_ij = (group[max_ij[0]], max_ij[1])
            if low_nmi[max_ij] < _COL_GRP_MIN_NMI:
                return group
            new_bins = bins * self.columns[max_ij[1]].bin_vals.size
            if new_bins > _COL_GRP_MAX_BIN:
                return group
            # prevent the new member be selected in the coming iterations
            bins = new_bins
            group.append(max_ij[1])
            low_nmi[:, max_ij[1]] = -1

        return group

    def _add_col_group(self, group):
        """Add a column group."""
        self.col_groups.append(group)

        if group[0] != self.label_col_idx:
            shape = [self.columns[self.label_col_idx].bin_vals.size]
        else:
            shape = []
        shape.extend([self.columns[i].bin_vals.size for i in group])
        joint_dist = np.zeros(shape, dtype=float)
        rec_count = self._bin_idxs[0].size
        mask = np.empty(rec_count, dtype=bool)

        for bi in range(np.prod(shape)):
            bv_idxs = np.unravel_index(bi, shape)
            mask.fill(True)
            for gi, bvi in enumerate(bv_idxs):
                if self.label_col_idx >= 0:
                    col_idx = self.label_col_idx if gi == 0 else group[gi-1]
                else:
                    col_idx = group[gi]
                mkey = f'mask_{col_idx}.{bvi}'
                col_mask = self._cache.get(mkey, None)
                if col_mask is None:
                    col_mask = (self._bin_idxs[col_idx] == bvi)
                    self._cache[mkey] = col_mask
                mask &= col_mask
            joint_dist[tuple(bv_idxs)] = np.sum(mask) / rec_count
        self.col_group_dists.append(joint_dist)


class TabSim:
    """
    Tabular data simulator.

    Args:
        tab_digest (TabDigest): Table digest.
        batch_size (int): Batch size for row data generation. Default: 10000

    Raises:
        TypeError: Be raised for any argument or input type problem.
        ValueError: Be raised for any input value problem.
    """
    def __init__(self, tab_digest, batch_size=10000):
        self._tab_digest = tab_digest
        self._batch_size = batch_size

    def generate(self, num_rows, writer, noise=0):
        """
        Generate simulated data.

        Args:
            num_rows (int): Total number of rows to be generated.
            writer (TabWriter): Writer for saving the generated data.
            noise (float): Noise for data's distributions, the interpolation weight between the original distributions
                in the table digest and pure uniform distributions. 0 means 100% follows the original distributions.
                Default: 0.
        """
        if noise < 0 or noise > 1:
            raise ValueError('Argument "noise" must be in range of [0.0, 1.0].')

        batch_count = num_rows // self._batch_size
        if num_rows > batch_count * self._batch_size:
            batch_count += 1

        schema = [(c.name, c.dtype) for c in self._tab_digest.columns]
        writer.begin(schema, self._batch_size, num_rows)

        col_count = len(self._tab_digest.columns)

        for bi in range(batch_count):
            gen_size = min(self._batch_size, num_rows - bi*self._batch_size)
            if self._tab_digest.label_col_idx >= 0:
                batch = self._gen_with_label_col(gen_size, noise)
            else:
                batch = [None] * col_count
                for group, dist in zip(self._tab_digest.col_groups, self._tab_digest.col_group_dists):
                    self._gen_col_grp_data(group, dist, gen_size, noise, batch)

            for i, col_data in enumerate(batch):
                if col_data is None:
                    raise RuntimeError(f'Column[{i}] is not generated.')
            writer.write(batch)

        writer.end()

    def _gen_with_label_col(self, gen_size, noise):
        """Generate simulated data with label column."""
        col_count = len(self._tab_digest.columns)
        batch = [None] * col_count
        bin_idxs = [None] * col_count

        for col in self._tab_digest.columns:
            if col.idx != self._tab_digest.label_col_idx:
                batch[col.idx] = np.empty(gen_size, dtype=col.dtype)

        self._gen_col_grp_data(self._tab_digest.col_groups[0],
                               self._tab_digest.col_group_dists[0],
                               gen_size, noise, batch, bin_idxs)

        label_bin_idxs = bin_idxs[self._tab_digest.label_col_idx]
        label_col = self._tab_digest.columns[self._tab_digest.label_col_idx]
        for label_bvi in range(label_col.bin_count):
            mask = (label_bin_idxs == label_bvi)
            segment_size = np.sum(mask)
            if segment_size == 0:
                continue
            segment = [None] * segment_size
            groups = self._tab_digest.col_groups[1:]
            dists = self._tab_digest.col_group_dists[1:]

            for group, dist in zip(groups, dists):
                self._gen_col_grp_data(group, dist[label_bvi], segment_size, noise, segment)
                for ci in group:
                    batch[ci][mask] = segment[ci]

        return batch

    def _gen_col_grp_data(self, group, dist, num_rows, noise, out_batch, out_bin_idxs=None):
        """Generate simulated data for a column group."""
        probs = dist.flatten()
        probs /= np.sum(probs)
        if noise > _EPS:
            probs = (1 - noise)*probs + noise/probs.size
            probs /= np.sum(probs)

        choices = np.arange(probs.size, dtype=int)
        keys = np.random.choice(choices, num_rows, p=probs)
        bin_idxs = np.unravel_index(keys, dist.shape)
        for i, ci in enumerate(group):
            if out_batch[ci] is not None:
                raise RuntimeError(f'Column[{ci}] is already generated.')
            col = self._tab_digest.columns[ci]
            out_batch[ci] = self._bins_to_vals(col, bin_idxs[i])
            if out_bin_idxs is not None:
                out_bin_idxs[ci] = bin_idxs[i]

    @staticmethod
    def _bins_to_vals(col, bin_idxs):
        """Convert bin indices to actual values."""
        if col.is_numeric:
            if col.dtype is int:
                if col.bin_width > 1:
                    offsets = np.random.randint(col.bin_width, size=bin_idxs.size)
                else:
                    return col.bin_vals[bin_idxs]
            else:
                if col.bin_width == 0:
                    return col.bin_vals[bin_idxs]
                offsets = np.random.uniform(0, col.bin_width, size=bin_idxs.shape[0])
            return col.bin_vals[bin_idxs] + offsets

        return col.bin_vals[bin_idxs]
