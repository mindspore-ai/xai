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
import subprocess
import shlex

import sklearn.datasets

if __name__ == "__main__":
    iris = sklearn.datasets.load_iris()
    features = iris.data
    labels = iris.target
    # save the dataset to file
    header = 'sepal_length|float,sepal_width|float,petal_length|float,petal_width|float,*class|cat'
    with open('real_table.csv', 'w') as f:
        f.write(header + '\n')
        for i in range(len(labels)):
            for feat in features[i]:
                f.write("{},".format(feat))
            f.write("{}\n".format(labels[i]))

    # digestion
    dig_cmd = "mindspore_xai tabdig real_table.csv digest.json"
    subprocess.call(shlex.split(dig_cmd))

    # simulation
    sim_cmd = "mindspore_xai tabsim digest.json sim_table.csv 200000"
    subprocess.call(shlex.split(sim_cmd))
