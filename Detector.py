import argparse
import subprocess
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt
import difflib


def pair_weiss_distance_diff(branch1_num, branch2_num):
    if branch1_num == branch2_num:
        return 1
    branch1 = branches[int(branch1_num[0])]
    branch2 = branches[int(branch2_num[0])]

    def dist(file_num1, file_num2, mode):
        pcounter = 0
        mcounter = 0
        if mode == '.ipynb':
            nbdiff = subprocess.check_output(('nbdiff ' +
                                              branch1_ipynb_files[file_num1] + ' ' +
                                              branch2_ipynb_files[file_num2] + " -M").split(' ')).decode("utf-8").split(
                '\n')
            for s in nbdiff:
                if len(s) > 5:
                    if s[5] == '+':
                        pcounter += 1
                    elif s[5] == '-':
                        mcounter += 1

        else:
            d = difflib.Differ()
            diff = d.compare(open(branch1_py_files[file_num1]).readlines(),
                      open(branch2_py_files[file_num2]).readlines())
            for s in diff:
                if len(s) > 2:
                    if s[0] == '+':
                        pcounter += 1
                    elif s[0] == '-':
                        mcounter += 1
        return pcounter, mcounter

    branch1_py_files = branches_py_files[branch1]
    branch1_ipynb_files = branches_ipynb_files[branch1]
    branch2_py_files = branches_py_files[branch2]
    branch2_ipynb_files = branches_ipynb_files[branch2]

    branch1_py_files_lens = branches_py_files_lens[branch1]
    branch1_ipynb_files_lens = branches_ipynb_files_lens[branch1]
    branch2_py_files_lens = branches_py_files_lens[branch2]
    branch2_ipynb_files_lens = branches_ipynb_files_lens[branch2]

    #     branch1_py_files_lens = list(map(take_len, branch1_py_files))
    #     branch1_ipynb_files_lens = list(map(take_len, branch1_ipynb_files))
    #     branch2_py_files_lens = list(map(take_len, branch2_py_files))
    #     branch2_ipynb_files_lens = list(map(take_len, branch2_ipynb_files))

    persent_of_same_code_nb = []
    if len(branch1_ipynb_files) != 0 and len(branch2_ipynb_files) != 0:
        for i in range(len(branch1_ipynb_files)):
            comonness = []
            for j in range(len(branch2_ipynb_files)):
                pcounter, mcounter = dist(i, j, '.ipynb')
                common_str_num = (branch1_ipynb_files_lens[i] - pcounter +
                                  branch2_ipynb_files_lens[j] - mcounter)
                if (branch1_ipynb_files_lens[i] + branch2_ipynb_files_lens[j]) == 0:
                    comonness.append(0)
                else:
                    comonness.append(common_str_num / (branch1_ipynb_files_lens[i] + branch2_ipynb_files_lens[j]))
            persent_of_same_code_nb.append(max(comonness))

    persent_of_same_code_py = []
    if len(branch1_py_files) != 0 and len(branch2_py_files) != 0:
        for i in range(len(branch1_py_files)):
            comonness = []
            for j in range(len(branch2_py_files)):
                pcounter, mcounter = dist(i, j, '.py')
                common_str_num = (branch1_py_files_lens[i] - pcounter +
                                  branch2_py_files_lens[j] - mcounter)
                if (branch1_py_files_lens[i] + branch2_py_files_lens[j]) == 0:
                    comonness.append(0)
                else:
                    comonness.append(common_str_num / (branch1_py_files_lens[i] + branch2_py_files_lens[j]))
            persent_of_same_code_py.append(max(comonness))

    if np.sum(branch1_ipynb_files_lens) == 0:
        sameness_nb = 0
    else:
        sameness_nb = np.dot(persent_of_same_code_nb, branch1_ipynb_files_lens) / np.sum(branch1_ipynb_files_lens)
    if np.sum(branch1_py_files_lens) == 0:
        sameness_py = 0
    else:
        sameness_py = np.dot(persent_of_same_code_py, branch1_py_files_lens) / np.sum(branch1_py_files_lens)
    sameness = ((sameness_nb * sum(branch1_ipynb_files_lens) + sameness_py * sum(branch1_py_files_lens)) /
                (sum(branch1_ipynb_files_lens) + sum(branch1_py_files_lens)))

    return sameness


def take_len(file):
    if '.ipynb' in file:
        return len(subprocess.check_output(('nbshow ' + file +
                                            " -M").split(' ')).decode("utf-8").split('\n'))
    else:
        return int(subprocess.check_output(('wc -l ' + file
                                            ).split(' ')).decode("utf-8").split(' ')[0])


def pair_weiss_distance_set(branch1_num, branch2_num):
    if branch1_num == branch2_num:
        return 1

    branch1 = branches[int(branch1_num[0])]
    branch2 = branches[int(branch2_num[0])]

    branch1_set = branches_text[branch1]
    branch2_set = branches_text[branch2]
    return len(branch1_set.intersection(branch2_set)) / len(branch1_set)

parser = argparse.ArgumentParser()
parser.add_argument('-u', action='store', dest='url',
                    help='Git repo URL',
                    default='None')
parser.add_argument('-f', action='store_true', default=False,
                    dest='mode',
                    help='Use full mode')

results = parser.parse_args()
url = results.url
mode = results.mode

if url == "None":
	print("")
	
assert url != "None", "There is no git url"

if os.path.isdir("work_folder"):
    os.system('rm -r -f work_folder')

assert os.system("git clone " + url + ' work_folder/master') == 0, "Git clone ERROR"

f = open('work_folder/master/branch.sh', 'w+')
f.write(
    '''
    cd work_folder/master
    git branch -a
    ''')
f.close()

branches = subprocess.check_output("bash work_folder/master/branch.sh".split()).decode("utf-8").split('\n')
branches = [branch.split('/')[-1] for branch in branches if "master" not in branch and branch != '']

if len(branches) == 1:
    assert False

for branch in branches:
    subprocess.check_output(("git clone " + url +
                             ' work_folder/' + branch + '/ ' +
                             '--branch ' + branch).split())

branches = ['master'] + branches

branches_text = {}
branches_py_files = {}
branches_ipynb_files = {}
branches_ipynb_files_lens = {}
branches_py_files_lens = {}

for branch in branches:
    branch_py_files = []
    for root, dirs, files in os.walk('work_folder/' + branch):
        py = [s for s in files if '.py' in s]
        branch_py_files += [root + '/' + s for s  in py]
    branches_py_files[branch] = branch_py_files

    branch_ipynb_files = []
    for root, dirs, files in os.walk('work_folder/' + branch):
        ipynb = [s for s in files if '.ipynb' in s]
        branch_ipynb_files += [root + '/' + s for s  in ipynb]
    branches_ipynb_files[branch] = branch_ipynb_files

if mode == False:
    for branch in branches:
        common_py_code = []
        for file in branches_py_files[branch]:
            f = open(file)
            common_py_code = common_py_code + list(f.readlines())
            f.close()

        common_ipynb_code = []
        for file in branches_ipynb_files[branch]:
            ipynb_code = subprocess.check_output(('nbshow ' + file +
                                                  " -M -A -O").split(' ')).decode("utf-8").split('\n')[1:]
            ipynb_code = [s.replace(' ', '') for s in ipynb_code if "source:" not in s and
                          "execution_count:" not in s and
                          'code cell' not in s]
            ipynb_code = [s for s in ipynb_code if s != '']
            common_ipynb_code += ipynb_code
        common_code = common_py_code + common_ipynb_code
        branches_text[branch] = set(common_code)

    matr = pairwise_distances(np.arange(len(branches)).reshape((-1, 1)),
                              metric=pair_weiss_distance_set,
                              n_jobs=-1)
else:
    for branch in branches:
        branch_py_files_lens = list(map(take_len, branches_py_files[branch]))
        branches_py_files_lens[branch] = branch_py_files_lens
        branch_ipynb_files_lens = list(map(take_len, branches_ipynb_files[branch]))
        branches_ipynb_files_lens[branch] = branch_ipynb_files_lens

    matr = pairwise_distances(np.arange(len(branches)).reshape((-1, 1)),
                              metric=pair_weiss_distance_diff,
                              n_jobs=1)

d = sch.distance.pdist(matr)
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(matr[np.argsort(ind), :][:, np.argsort(ind)], ax=axes, annot = True,
            yticklabels=np.array(branches)[np.argsort(ind)],
            xticklabels=np.array(branches)[np.argsort(ind)])
plt.title("Mesure of similarity")
plt.savefig('result.pdf')
plt.show()
