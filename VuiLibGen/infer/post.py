    #!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import json
import time, datetime
import pandas as pd


import Levenshtein

# Maven_path is the path to the  `maven_corpus.json`, as the descriptions of maven packages.
maven_path = ""  # Path to maven dataset
with open(maven_path, 'r') as f:
    maven_corpus = json.load(f)

weights = (1, 2, 2)

# Search the cloest suffix in the dict with input suffix
def cloest_artifact(artifact_id):
    global artifacts, weights
    if artifact_id in artifacts:
        return artifact_id

    distances = [(Levenshtein.distance(artifact_id, item,\
                    weights = weights), item) for item in artifacts]
    return min(distances)[1]

# Search for the cloest prefix in the group of all corresponding prefix with the known suffix
def cloest_group(group_id, groups):
    if len(groups) == 0:
        return group_id
    if len(groups) == 1:
        return next(iter(groups))
    
    global weights
    distances = [(Levenshtein.distance(group_id, item.split(':')[-2],\
                    weights = weights), item) for item in groups]
    return min(distances)[1]


# Local search Algorithm
def closest_lib(label: str) -> str:
    global lib_names
    if label in lib_names:
        return label
    if len(label.split(':')) > 1:
        group_id, artifact_id = label.split(':')[-2], label.split(':')[-1]
    else:
        group_id, artifact_id = "", label.split(':')[-1]
    if artifact_id in artifacts:
        return cloest_group(group_id, artifacts[artifact_id])
    else:
        advanced_artifact_id = cloest_artifact(artifact_id)
        return cloest_group(group_id, artifacts[advanced_artifact_id])
    
#Ground truth of library name 
lib_names = set([lib['name'] for lib in maven_corpus])

# Artifacts is a dictionary that maps a suffix into the its correspondings prefix
artifacts = {item.split(':')[-1]: set() for item in lib_names}
for item in lib_names:
    components = item.split(':')
    artifacts[components[-1]].add(item)
    
if __name__ == '__main__':
    # st_time = datetime.datetime.now()
    # restrict_lib = len(sys.argv) > 4 and sys.argv[4][0] == 'r'
    # post_process(test_path = sys.argv[1], output_path = sys.argv[2], maven_path = sys.argv[3], restrict_lib = restrict_lib)
    # end_time = datetime.datetime.now()
    # print(sys.argv[1], st_time, end_time, end_time - st_time)
    print(sys.argv[1])

