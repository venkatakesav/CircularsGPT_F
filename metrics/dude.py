# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DUDE dataset loader"""


from datasets import load_dataset

from codetiming import Timer

for binding in ["dict_annotations (new)"]: #"dict_PDF", 
    with Timer(name=f"{binding}", text=binding + " Elapsed time: {:.4f} seconds"):
        if binding == "dict_annotations (new)":
            ds = load_dataset("jordyvl/DUDE_loader", 'Amazon_original', data_dir="./DUDE_train-val-test_binaries") #ignore_verifications=True, , writer_batch_size=10
        else:
            ds = load_dataset("jordyvl/DUDE_loader", revision='db20bbf751b14e14e8143170bc201948ef5ac83c')

import pdb; pdb.set_trace()  # breakpoint d45ace65 //