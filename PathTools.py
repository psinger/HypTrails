'''
Created on 25.02.2013

@author: psinger
'''

import csv
from random import choice
from random import shuffle

def id_to_category(paths, page_categories, categories, filename):
    '''
    Transform paths with ids to paths with categories
    paths = file with paths
    page_categories = file with assignments of page ids to categories
    categories = meta category information
    filename = output filename for transformed paths
    '''
    cats = {}
    with open(categories) as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            cats[line[1]] = line[0]
    print cats
    id2cats = {}
    curr_id = ""
    with open(page_categories) as f:
        tmp_cats = []
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if line[0] == "-1":
                if curr_id != "":
                    id2cats[curr_id] = tmp_cats
                curr_id = line[1]
                tmp_cats = []
            else:
                tmp_cats.append(cats[line[2]])    
    
    cats_reverse = dict([(v, k) for k, v in cats.iteritems()])
    
    print id2cats["3522791"]
    print len(id2cats)
    print "====="
    tmp_paths = []
    done = []
    with open(paths) as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            for x in line:
                if x == "791" or x == "6125" or x == "11867" or x == "2462183" or x == "147575":
                    if x not in done:
                        cat_choice = id2cats.get(x, "0")#choice(id2cats.get(x, "0"))
                        print x, cat_choice, [cats_reverse[y] for y in cat_choice]#, cats_reverse[cat_choice]
                        done.append(x)
                    
                    
            tmp_paths.append([choice(id2cats.get(x, "0")) for x in line])
    """        
    with open(filename, "w") as text_file:
        for line in tmp_paths:
            text_file.write("\t".join(line)+"\n")
    """      
def id_to_category_count(paths, page_categories, categories, filename):
    '''
    Transform paths with ids to paths with the number of categories the page is assigned to
    paths = file with paths
    page_categories = file with assignments of page ids to categories
    categories = meta category information
    filename = output filename for transformed paths
    '''
    cats = {}
    with open(categories) as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            cats[line[1]] = line[0]
    print cats
    id2cats = {}
    curr_id = ""
    with open(page_categories) as f:
        tmp_cats = []
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if line[0] == "-1":
                if curr_id != "":
                    id2cats[curr_id] = tmp_cats
                curr_id = line[1]
                tmp_cats = []
            else:
                tmp_cats.append(cats[line[2]])    
            
    print id2cats["3522791"]
    print len(id2cats)
    
    tmp_paths = []
    with open(paths) as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            tmp_paths.append([len(id2cats.get(x, [])) for x in line])
            
    with open(filename, "w") as text_file:
        for line in tmp_paths:
            text_file.write("\t".join([str(x) for x in line])+"\n")
            
def stratified_kfold(paths, k=10):
    '''
    Performs stratified kfold on a set of paths
    paths = list of paths (each row is a numpy array of elements of a path)
    k = number of folds
    '''
    total_sum = sum([len(x) for x in paths])
    print "total nr. of paths: ", len(paths)
    print "total_sum= ", total_sum
    print "==========="
    wanted_sum = float(total_sum) / float(k)
    shuffle(paths)
    
    folds = []
    curr_fold = []
    curr_len = 0
    for line in paths:
        curr_fold.append(line)
        curr_len += len(line)
        if curr_len >= wanted_sum:
            folds.append(curr_fold)
            curr_fold =  []
            curr_len = 0
            
    if curr_len >= 0:
        folds.append(curr_fold)
        
    assert(len(folds) == k)
    
    return folds
                
#    for f in folds:
#        print "nr of paths: ", len(f)
#        print "click sum: ", sum([len(x) for x in f])
#        print "-------------"
            
