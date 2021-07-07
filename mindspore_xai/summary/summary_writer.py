# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Explanation summary writer."""
import os
import time

from mindspore.train._utils import check_value_type
from mindspore.train.summary._summary_adapter import get_event_file_name
from mindspore.train.summary.writer import BaseWriter
from mindspore.train._utils import _make_directory

from mindspore_xai.proto.summary_pb2 import Event, Explain


class SummaryWriter:
    """Explanation summary writer."""

    def __init__(self, summary_dir):
        self._timestamp = int(time.time())
        self._summary_dir = _make_directory(summary_dir)
        self._filename = get_event_file_name('events', '_explain', self._timestamp)
        self._path = os.path.join(self._summary_dir, self._filename)
        self._writer_impl = None
        self._closed = False

    def __enter__(self):
        """Enter the context manager."""
        if self._closed:
            raise RuntimeError(f'{self.__class__.__name__} was closed.')
        return self

    def __exit__(self, *err):
        """Exit the context manager."""
        del err
        self.close()

    @property
    def summary_dir(self):
        return self._summary_dir

    @property
    def filename(self):
        return self._filename

    @property
    def path(self):
        return self._path

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def closed(self):
        return self._closed

    def write(self, explain):
        """
        Write an explain data to summary.

        Args:
            explain (Explain): An Explain object from summary_pb2.

        Raises:
            RuntimeError: Be raised for writer was closed already.
            TypeError: Be raised for data is not an Explain object.
        """
        check_value_type("explain", explain, Explain)

        if self._closed:
            raise RuntimeError(f'{self.__class__.__name__} was closed.')

        event = Event()
        event.step = 1
        event.wall_time = time.time()
        event.explain.ParseFromString(explain.SerializeToString())

        if self._writer_impl is None:
            self._writer_impl = BaseWriter(self._path)
        self._writer_impl.write(plugin='explainer', data=event.SerializeToString())

    def close(self):
        """Close the writer."""
        if self._writer_impl is not None:
            self._writer_impl.close()
            self._writer_impl = None
        self._closed = True
