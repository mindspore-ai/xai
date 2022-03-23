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
"""Relation path data generator."""
import io
import random
import json
import csv
import warnings


class _UserRec:
    """User record, helper class for path generation."""
    def __init__(self, src_id, intern_id):
        self.src_id = src_id
        self.intern_id = intern_id
        self.positive_items = dict()
        self.interact_items = dict()
        self.other_items = dict()
        self.has_unseen_ref = False

    def add_item(self, item_rec, rating):
        if rating == 'p':
            item_dict = self.positive_items
        elif rating == 'c':
            item_dict = self.interact_items
        else:
            item_dict = self.other_items
        item_dict[item_rec.intern_id] = item_rec


class _ItemRec:
    """Item record, helper class for path generation."""
    def __init__(self, src_id, intern_id, ref_src_ids, ref_ids):
        self.src_id = src_id
        self.intern_id = intern_id
        self.ref_src_ids = ref_src_ids
        self.ref_ids = ref_ids


def reverse_id_maps(id_maps):
    """
    Reverse the key-value position of object id maps.

    Args:
        id_maps(dict[any, dict[any, any]]): Collection of object id maps.

    Returns:
        dict[any, dict[any, any]], collection of the reversed object id maps.

    Examples:
        >>> import io
        >>> import json
        >>> from mindspore_xai.whitebox.tbnet import reverse_id_maps
        >>>
        >>> with io.open("id_maps.json", mode="r", encoding="utf-8") as f:
        >>>     id_maps = json.load(f)
        >>> reversed_id_maps = reverse_id_maps(id_maps)
    """
    reversed_maps = dict()
    for obj_type, id_map in id_maps.items():
        if not isinstance(id_map, dict):
            continue
        reversed_map = dict()
        for src_id, intern_id in id_map.items():
            reversed_map[intern_id] = src_id
        reversed_maps[obj_type] = reversed_map
    return reversed_maps


class PathGen:
    """
    Generate relation path csv from the source csv table.

    Args:
        per_item_paths (int): Number of relation paths per subject item, must be positive.
        same_relation (bool): True to only generate paths that relation1 is same as relation2, usually faster.
        id_maps (dict[str, Union[dict[str, int], int]], Optional): Object id maps, the internal id baseline, new user,
            item and entity IDs will be based on that. If Which is None or empty, grow_id_maps will be True by
            default.

    Examples:
        >>> import io
        >>> import json
        >>> from mindspore_xai.whitebox.tbnet import PathGen
        >>>
        >>> ################ single source csv, single output csv ################
        >>> path_gen = PathGen(per_item_paths=24)
        >>> path_gen.generate("src_all.csv", "out_paths.csv")
        >>>
        >>> ################ multiple source csv, single output csv ################
        >>> path_gen = PathGen(per_item_paths=24)
        >>> with io.open("out_paths2.csv", mode="w") as out_csv:
        >>>     path_gen.generate("src_user_0001_1000.csv", out_csv)
        >>>     path_gen.generate("src_user_1001_2000.csv", out_csv)
        >>>
        >>> ################ multiple source csv, multiple output csv ################
        >>> path_gen = PathGen(per_item_paths=24)
        >>> path_gen.generate("src_user_0001_1000.csv", "out_paths_part1.csv")
        >>> path_gen.generate("src_user_1001_2000.csv", "out_paths_part2.csv")
        >>>
        >>> ################ resume the path generation ################
        >>> with io.open("id_maps.json", mode="r", encoding="utf-8") as f:
        >>>     id_maps = json.load(f)
        >>> path_gen = PathGen(per_item_paths=24, id_maps=id_maps)
        >>> # for adding new items and ref_ids' internal ID, otherwise, 0 will be assigned.
        >>> path_gen.grow_id_maps = True
        >>> # The ID of new user, item and reference will be continued from part1
        >>> path_gen.generate("src_part2.csv", "out_part2.csv")
        >>>
        >>> ################ generate query csv for inference ################
        >>> path_gen = PathGen(per_item_paths=24, id_maps=id_maps)
        >>> # DO NOT set grow_id_maps to True for assigning 0 as internal ID of the unseen items and references
        >>> # the generated csv will only contains candidate items with rating 'c' or 'x'
        >>> path_gen.subject_ratings = "cx"
        >>> # generate single user query csv for infer script
        >>> path_gen.generate("src_user_0001.csv", "query_user_0001.csv")
        >>>
        >>> ################ save id maps ################
        >>> with io.open("id_maps.json", mode="w", encoding="utf-8") as f:
        >>>     json.dump(path_gen.id_maps(), f, indent=4)
    """
    def __init__(self, per_item_paths, same_relation=False, id_maps=None):

        self._per_item_paths = per_item_paths
        self._same_relation = same_relation

        self._user_id_counter = 1
        self._entity_id_counter = 1
        self._num_relations = 0
        self._rows_generated = 0
        self._user_rec = None

        if id_maps:
            self._item_id_map = id_maps.get('item', dict())
            self._ref_id_map = id_maps.get('reference', dict())
            self._rl_id_map = id_maps.get('relation', None)
            self._user_id_counter = id_maps.get('_user_id_counter', self._user_id_counter)
            max_item_id = max(self._item_id_map.values()) if self._item_id_map else 0
            max_ref_id = max(self._ref_id_map.values()) if self._ref_id_map else 0
            self._entity_id_counter = max(max_item_id, max_ref_id) + 1
        else:
            self._item_id_map = dict()
            self._ref_id_map = dict()
            self._rl_id_map = None

        self.grow_id_maps = not (bool(self._item_id_map) and bool(self._ref_id_map))
        self.subject_ratings = None

        self._unseen_items = 0
        self._unseen_refs = 0

    @property
    def num_users(self):
        """No. of distinct users."""
        return self._user_id_counter - 1

    @property
    def num_references(self):
        """No. of distinct references."""
        return len(self._ref_id_map)

    @property
    def num_items(self):
        """No. of distinct items."""
        return len(self._item_id_map)

    @property
    def num_relations(self):
        """No. of distinct relations."""
        return self._num_relations

    @property
    def rows_generated(self):
        """Total no. of rows generated to the output CSVs."""
        return self._rows_generated

    @property
    def per_item_paths(self):
        """No. of path per subject item."""
        return self._per_item_paths

    @property
    def same_relation(self):
        """Only generate paths with the same relation on both sides?"""
        return self._same_relation

    @property
    def unseen_items(self):
        """Total no. of unseen items has encountered."""
        return self._unseen_items

    @property
    def unseen_refs(self):
        """Total no. of unseen references has encountered."""
        return self._unseen_refs

    def id_maps(self):
        """Object ID maps."""
        maps = {
            "item": dict(self._item_id_map),
            "reference": dict(self._ref_id_map),
            "_user_id_counter": self._user_id_counter
        }
        if self._rl_id_map is not None:
            maps["relation"] = dict(self._rl_id_map)
        return maps

    def generate(self, in_csv, out_csv, in_sep=',', in_mv_sep=';', in_encoding='utf-8'):
        """
        Generate paths csv from the source CSV files.

        args:
            in_csv (Union[str, TextIOBase]): The input source csv path or stream.
            out_csv (Union[str, TextIOBase]): The output source csv path or stream.
            in_sep (str): Separator of the input csv.
            in_mv_sep (str): Multi-value separator of the input csv.
            in_encoding (str): Encoding of the input source csv, ingored if in_csv is a text stream already.

        Returns:
            int, no. of rows that generated to the output csv in this call.
        """
        if not isinstance(in_csv, (str, io.TextIOBase)):
            raise TypeError(f"Unexpected in_csv type:{type(in_csv)}")
        if not isinstance(out_csv, (str, io.TextIOBase)):
            raise TypeError(f"Unexpected out_csv type:{type(out_csv)}")

        opened_files = []
        try:
            if isinstance(in_csv, str):
                in_csv = io.open(in_csv, mode="r", encoding=in_encoding)
                opened_files.append(in_csv)
            in_csv = csv.reader(in_csv, delimiter=in_sep)
            col_indices = self._pre_generate(in_csv, None)

            if isinstance(out_csv, str):
                out_csv = io.open(out_csv, mode="w", encoding="ascii")
                opened_files.append(out_csv)
            rows_generated = self._do_generate(in_csv, out_csv, in_mv_sep, col_indices)

        except (IOError, ValueError, RuntimeError, PermissionError, KeyError) as e:
            raise e
        finally:
            for f in opened_files:
                f.close()
        return rows_generated

    def _pre_generate(self, in_csv, in_col_map):
        """Prepare for the path generation."""
        if in_col_map is not None:
            expected_cols = self._default_abstract_header(len(in_col_map) - 3)
            map_values = list(in_col_map.values())
            for col in expected_cols:
                if col not in map_values:
                    raise ValueError("col_map has no '{col}' value.")

        header = self._read_header(in_csv)
        if len(header) < 4:
            raise IOError(f"No. of in_csv columns:{len(header)} is less than 4.")
        num_relations = len(header) - 3
        if self._num_relations > 0:
            if num_relations != self._num_relations:
                raise IOError(f"Inconsistent no. of in_csv relations.")
        else:
            self._num_relations = num_relations

        col_indices = self._get_col_indices(header, in_col_map)
        rl_id_map = self._to_relation_id_map(header, col_indices)

        if not self._rl_id_map:
            self._rl_id_map = rl_id_map
        elif rl_id_map != self._rl_id_map:
            raise IOError(f"Inconsistent in_csv relations.")

        return col_indices

    def _do_generate(self, in_csv, out_csv, in_mv_sep, col_indices):
        """Do generate the paths."""
        old_rows_generated = self._rows_generated
        old_unseen_items = self._unseen_items
        old_unseen_refs = self._unseen_refs

        col_count = len(col_indices)
        self._user_rec = None
        for line in in_csv:
            values = list(map(lambda x: x.strip(), line))
            if len(values) != col_count:
                raise IOError(f"No. of in_csv columns:{len(values)} is not {col_count}.")
            self._process_line(values, in_mv_sep, col_indices, out_csv)

        if self._user_rec is not None:
            self._process_user_rec(self._user_rec, out_csv)
            self._user_rec = None

        delta_unseen_items = self._unseen_items - old_unseen_items
        delta_unseen_refs = self._unseen_refs - old_unseen_refs
        if delta_unseen_items > 0:
            warnings.warn(f"{delta_unseen_items} unseen items' internal IDs were set to 0, "
                          f"set grow_id_maps to True for adding new internal IDs.", RuntimeWarning)
        if delta_unseen_refs > 0:
            warnings.warn(f"{delta_unseen_refs} unseen references' internal IDs were set to 0, "
                          f"set grow_id_maps to True for adding new internal IDs.", RuntimeWarning)

        return self._rows_generated - old_rows_generated

    def _process_line(self, values, in_mv_sep, col_indices, out_csv):
        """Process a line from the input CSV."""
        user_src = values[col_indices[0]]
        item_src = values[col_indices[1]]
        rating = values[col_indices[2]].lower()
        if rating not in ('p', 'c', 'x'):
            raise IOError(f"Unrecognized rating:'{rating}', must be one of 'p', 'c' or 'x'.")
        ref_srcs = [values[col_indices[i]] for i in range(3, len(col_indices))]

        if in_mv_sep:
            ref_srcs = list(map(lambda x: list(map(lambda y: y.strip(), x.split(in_mv_sep))), ref_srcs))
        else:
            ref_srcs = list(map(lambda x: [x], ref_srcs))

        if self._user_rec is not None and user_src != self._user_rec.src_id:
            # user changed
            self._process_user_rec(self._user_rec, out_csv)
            self._user_rec = None

        if self._user_rec is None:
            self._user_rec = _UserRec(user_src, self._user_id_counter)
            self._user_id_counter += 1

        item_rec, has_unseen_ref = self._to_item_rec(item_src, ref_srcs)
        self._user_rec.add_item(item_rec, rating)
        self._user_rec.has_unseen_ref |= has_unseen_ref

    def _process_user_rec(self, user_rec, out_csv):
        """Generate paths for an user."""
        positive_count = 0

        subject_items = []

        if not self.subject_ratings:
            subject_items.extend(user_rec.positive_items.values())
            subject_items.extend(user_rec.other_items.values())
            positive_count = len(user_rec.positive_items)
        else:
            if 'p' in self.subject_ratings:
                subject_items.extend(user_rec.positive_items.values())
                positive_count = len(user_rec.positive_items)
            if 'c' in self.subject_ratings:
                subject_items.extend(user_rec.interact_items.values())
            if 'x' in self.subject_ratings:
                subject_items.extend(user_rec.other_items.values())

        hist_items = []
        hist_items.extend(user_rec.positive_items.values())
        hist_items.extend(user_rec.interact_items.values())

        for i, subject in enumerate(subject_items):

            paths = []
            for hist in hist_items:
                if hist.src_id == subject.src_id:
                    continue
                self._find_paths(not user_rec.has_unseen_ref, subject, hist, paths)

            if not paths:
                continue

            paths = random.sample(paths, min(len(paths), self._per_item_paths))

            row = [0]*(2 + self._per_item_paths*4)
            row[0] = subject.intern_id               # subject item
            row[1] = 1 if i < positive_count else 0  # label
            for p, path in enumerate(paths):
                offset = 2 + p*4
                for j in range(4):
                    row[offset + j] = path[j]
            out_csv.write(','.join(map(str, row)))
            out_csv.write('\n')
            self._rows_generated += 1

    def _find_paths(self, by_intern_id, subject_item, hist_item, paths):
        """Find paths between the subject and historical item."""
        if by_intern_id:
            for i, ref_list in enumerate(subject_item.ref_ids):
                for ref in ref_list:
                    self._find_paths_by_intern_id(i, ref, hist_item, paths)
        else:
            for i, (ref_src_list, ref_list) in enumerate(zip(subject_item.ref_src_ids, 
                                                             subject_item.ref_ids)):
                for src_ref, ref in zip(ref_src_list, ref_list):
                    self._find_paths_by_src(i, src_ref, ref, hist_item, paths)

    def _find_paths_by_intern_id(self, subject_ridx, ref_id, hist_item, paths):
        """Find paths by internal reference ID, a bit faster."""
        if self._same_relation:
            if ref_id in hist_item.ref_ids[subject_ridx]:
                relation_id = self._ridx_to_relation_id(subject_ridx)
                paths.append((relation_id,
                              ref_id,
                              relation_id,
                              hist_item.intern_id))
        else:
            for hist_ridx, hist_ref_list in enumerate(hist_item.ref_ids):
                if ref_id in hist_ref_list:
                    paths.append((self._ridx_to_relation_id(subject_ridx),
                                  ref_id,
                                  self._ridx_to_relation_id(hist_ridx),
                                  hist_item.intern_id))

    def _find_paths_by_src(self, subject_ridx, ref_src_id, ref_id, hist_item, paths):
        """Find paths by source reference ID."""
        if self._same_relation:
            if ref_src_id in hist_item.ref_src_ids[subject_ridx]:
                relation_id = self._ridx_to_relation_id(subject_ridx)
                paths.append((relation_id,
                              ref_id,
                              relation_id,
                              hist_item.intern_id))
        else:
            for hist_ridx, hist_ref_src_list in enumerate(hist_item.ref_src_ids):
                if ref_src_id in hist_ref_src_list:
                    paths.append((self._ridx_to_relation_id(subject_ridx),
                                  ref_id,
                                  self._ridx_to_relation_id(hist_ridx),
                                  hist_item.intern_id))

    def _ridx_to_relation_id(self, idx):
        """Relation index to id."""
        return idx

    def _to_relation_id_map(self, header, col_indices):
        """Convert input csv header to a relation id map."""
        id_map = {}
        id_counter = 0
        for i in range(3, len(col_indices)):
            id_map[header[col_indices[i]]] = id_counter
            id_counter += 1
        if len(id_map) < len(header) - 3:
            raise IOError("Duplicated column!")
        return id_map

    def _to_item_rec(self, item_src, ref_srcs):
        """Convert the item src id and the source reference to an item record."""
        item_id = self._item_id_map.get(item_src, -1)
        if item_id == -1:
            if not self.grow_id_maps:
                item_id = 0
                self._unseen_items += 1
            else:
                item_id = self._entity_id_counter
                self._item_id_map[item_src] = item_id
                self._entity_id_counter += 1
               
        has_unseen_ref = False
        ref_ids = [[] for _ in range(len(ref_srcs))]
        for i, ref_src_list in enumerate(ref_srcs):
            for ref_src in ref_src_list:
                ref_id = self._ref_id_map.get(ref_src, -1)
                if ref_id == -1:
                    if not self.grow_id_maps:
                        ref_id = 0
                        self._unseen_refs += 1
                        has_unseen_ref = True
                    else:
                        ref_id = self._entity_id_counter
                        self._ref_id_map[ref_src] = ref_id
                        self._entity_id_counter += 1
                ref_ids[i].append(ref_id)

        return _ItemRec(item_src, item_id, ref_srcs, ref_ids), has_unseen_ref

    def _get_col_indices(self, header, col_map):
        """Find the column indices base on the mapping."""
        if col_map:
            mapped = [col_map[col] for col in header]
            default_header = self._default_abstract_header(len(header)-3)
            return [mapped.index(col) for col in default_header]
        return range(len(header))

    @staticmethod
    def _read_header(in_csv):
        """Read the CSV header."""
        line = next(in_csv)
        splited = list(map(lambda x: x.strip(), line))
        return splited

    @staticmethod
    def _default_abstract_header(num_relation):
        """Get the default abstract header."""
        abstract_header = ["user", "item", "rating"]
        abstract_header.extend([f"r{i + 1}" for i in num_relation])
        return abstract_header
