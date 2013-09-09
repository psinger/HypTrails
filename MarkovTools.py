'''
Created on 20.02.2013

@author: Killver
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import types
import copy
import scipy.sparse as sp    

def likelihood_ratio_test(l, p):
    '''
    Performing likelihood ratio test
    l = dictionary of likelihoods
    p = dictionary of parameter counts
    '''
    lratios = {}
    pvals = {}
    dfs = {}
    
    for null_k, null_l in l.iteritems():
        null_p = p[null_k]
        for test_k, test_l in l.iteritems():
            if test_k <= null_k and test_k != max(l.keys()):
                continue
            test_p = p[test_k]
            #likelihood ratio
            if null_k == test_k:
                lr = 0.0
            else:
                lr = -2*(null_l-test_l)
            #degrees of freedom
            df = test_p - null_p
            #print null_l, v, df
            p_val = 1 - stats.chi2.cdf(lr, df)
            #print p_val
            lratios[(null_k, test_k)] = lr
            pvals[(null_k, test_k)] = p_val
            dfs[(null_k, test_k)] = df
        
    return lratios, pvals, dfs

def akaike_information_criterion(lratios, dfs, null_model):
    '''
    Performing akaike information criterion (AIC)
    Needs to be called after likelihood_ratio_test
    lrts = dictionary of likelihood ratios
    dfs = dictionary of degrees of freedom
    null_model = the k value for the model you want to test against
    choose model with minimum AIC value
    '''
    aics = {}
    
    for k in lratios.keys():
        if k[1] != null_model:
            continue
        aic = lratios[k] - 2*(dfs[k])
        aics[k[0]] = aic
        
    return aics

def bayesian_information_criterion(lratios, dfs, obs, null_model):
    '''
    Performing bayesian information criterion (BIC)
    Needs to be called after likelihood_ratio_test
    lrts = dictionary of likelihood ratios
    dfs = dictionary of degrees of freedom
    obs = observation counts
    choose model with minimum BIC value
    '''
    bics = {}
    
    for k in lratios.keys():
        if k[1] != null_model:
            continue
        bic = lratios[k] - (dfs[k] * math.log(obs[k[0]]))
        bics[k[0]] = bic
        
    return bics

def bayesian_model_selection(evidences, params, penalty = False):
    '''
    Model selection using already calculated bayesian evidences for a set of models
    Needs to be called after bayesian_evidence
    evidences = dictionary of evidences
    params = dictionary of parameter counts
    penalty = add parameter penalty
    choose model with highest probability
    '''
    evidences_tmp = copy.deepcopy(evidences)
    model_probas = {}
    
    if penalty:
        for k in evidences.keys():
            evidences_tmp[k] -= params[k]
    
    max_evidence = max(evidences_tmp.values())
    denom = 0.0
    for k in evidences_tmp.keys():
        denom += np.exp(evidences_tmp[k] - max_evidence)

    denom = np.log(denom)
    denom += max_evidence
    
    for k,v in evidences_tmp.iteritems():
        proba = v - denom
        model_probas[k] = np.exp(proba)
        
    return model_probas

'''
LATEX AND PLOT HELPER FUNCTIONS
'''

def lrt_table_single(lratios, pvals, caption, label, filename, header=None):
    '''
    Provides latex table syntax given two dictionaries of lrt statistics
    Just prints lratios with statistical significance
    All dictionaries consisting of tuples as keys and values
    lratios = likelihood ratio values
    pvals = pvals (chi2)
    header = the header of the table, if None no header will be used
    '''
    string = ""
    string += "\\begin{tabular}[b]{|c|l|} \\hline"
    string += "\n"
    
    if header != None:
        string += " & ".join(header) + "\\\\ \\hline"
        string += "\n"
    
    for i in sorted(lratios.keys()):
        first = i[0]
        second = i[1]
        if first == second:
            continue
        model = "${_" + str(first) + "}\eta{_" + str(second) + "}$"
        #p = "%.10f" % 
        lr_string = str(round(lratios[i],4))
        if pvals[i] < 0.01:
            lr_string += "*"
        if pvals[i] < 0.001:
            lr_string += "*"
        string += model + " & " + lr_string + " \\\\ \\hline"
        string += "\n"
        
    string += "\\end{tabular}"
    string += "\n"
    string += "\\caption{" + caption + "}"
    string += "\n"
    string += "\\label{" + label + "}"
    
    print string
    
    with open(filename, "w") as text_file:
        text_file.write(string)

def lrt_table_single_combined(lratios1, pvals1, lratios2, pvals2, caption, label, filename, header=None):
    '''
    Provides latex table syntax given two dictionaries of lrt statistics
    FOR TWO DATASETS
    Just prints lratios with statistical significance
    All dictionaries consisting of tuples as keys and values
    lratios = likelihood ratio values
    pvals = pvals (chi2)
    header = the header of the table, if None no header will be used
    '''
    string = ""
    string += "\\begin{tabular}[b]{|c|l|l|} \\hline"
    string += "\n"
    
    if header != None:
        string += " & ".join(header) + "\\\\ \\hline"
        string += "\n"
    
    for i in sorted(lratios1.keys()):
        first = i[0]
        second = i[1]
        if first == second:
            continue
        model = "${_" + str(first) + "}\eta{_" + str(second) + "}$"
        #p = "%.10f" % 
        lr_string1 = str(round(lratios1[i],4))
        if pvals1[i] < 0.01:
            lr_string1 += "*"
        if pvals1[i] < 0.001:
            lr_string1 += "*"
        lr_string2 = str(round(lratios2[i],4))
        if pvals2[i] < 0.01:
            lr_string2 += "*"
        if pvals2[i] < 0.001:
            lr_string2 += "*"
        string += model + " & " + lr_string1 + " & " + lr_string2 + " \\\\ \\hline"
        string += "\n"
        
    string += "\\end{tabular}"
    string += "\n"
    string += "\\caption{" + caption + "}"
    string += "\n"
    string += "\\label{" + label + "}"
    
    print string
    
    with open(filename, "w") as text_file:
        text_file.write(string)

def lrt_table_trip(lratios1, pvals1, lratios2, pvals2, lratios3, pvals3, caption, label, filename, header=None):
    '''
    Provides latex table syntax given three dictionaries of lrt statistics
    FOR THREE DATASETS
    Just prints lratios with statistical significance
    All dictionaries consisting of tuples as keys and values
    lratios = likelihood ratio values
    pvals = pvals (chi2)
    header = the header of the table, if None no header will be used
    '''
    string = ""
    string += "\\begin{tabular}[b]{|c|l|l|l|} \\hline"
    string += "\n"
    
    if header != None:
        string += " & ".join(header) + "\\\\ \\hline"
        string += "\n"
    
    for i in sorted(lratios1.keys()):
        first = i[0]
        second = i[1]
        if first == second:
            continue
        model = "${_" + str(first) + "}\eta{_" + str(second) + "}$"
        #p = "%.10f" % 
        lr_string1 = ""
        if pvals1[i] < 0.01:
            lr_string1 += "*"
        if pvals1[i] < 0.001:
            lr_string1 += "*"
        lr_string2 = ""
        if pvals2[i] < 0.01:
            lr_string2 += "*"
        if pvals2[i] < 0.001:
            lr_string2 += "*"
        lr_string3 = ""
        if pvals3[i] < 0.01:
            lr_string3 += "*"
        if pvals3[i] < 0.001:
            lr_string3 += "*"
        string += model + " & " + lr_string1 + " & " + lr_string2 + " & " + lr_string3 + " \\\\ \\hline"
        string += "\n"
        
    string += "\\end{tabular}"
    string += "\n"
    string += "\\caption{" + caption + "}"
    string += "\n"
    string += "\\label{" + label + "}"
    
    print string
    
    with open(filename, "w") as text_file:
        text_file.write(string)

def lrt_table(lratios, pvals, dfs, caption, label, filename, header=None):
    '''
    Provides latex table syntax given three dictionaries of lrt statistics
    All dictionaries consisting of tuples as keys and values
    lratios = likelihood ratio values
    pvals = pvals (chi2)
    dfs = degree of freedoms
    header = the header of the table, if None no header will be used
    '''
    string = "\\begin{table}[ht!]"
    string += "\n"
    string += "\\centering"
    string += "\n"
    string += "\\caption{" + caption + "}"
    string += "\n"
    
    string += "\\begin{tabular}{|l|r|r|r} \\hline"
    string += "\n"
    if header != None:
        string += " & ".join(header) + "\\\\ \\hline"
        string += "\n"
    
    for i in sorted(lratios.keys()):
        first = i[0]
        second = i[1]
        if first == second:
            continue
        model = "${_" + str(first) + "}\eta{_" + str(second) + "}$"
        #p = "%.10f" % 
        string += model + " & " + str(round(lratios[i],4)) + " & " + str(dfs[i]) + " & %.6f \\\\ \\hline" % (round(pvals[i],6))
        string += "\n"
        
    string += "\\end{tabular}"
    string += "\n"
    string += "\\label{" + label + "}"
    string += "\n"
    string += "\\end{table}"
    
    print string
    
    with open(filename, "w") as text_file:
        text_file.write(string)

def plot_kvalues(l, xlabel, ylabel, filename, mark = "high"):
    '''
    Plotting the likelihoods given in dictionary
    mark = 'high' or 'low'
    '''
    plt.figure()
    plt.plot(l.keys(), l.values(), marker='o')
    
    markers_x = []
    markers_y = []
    for k,v in l.iteritems():
        if mark == "high":
            if v == max(l.values()):
                markers_x.append(k)
                markers_y.append(v)
        if mark == "low":
            if v == min(l.values()):
                markers_x.append(k)
                markers_y.append(v)
    plt.plot(markers_x, markers_y, 'rD')
    
    #print min(l.keys())
    ticks = np.arange(min(l.keys()), max(l.keys())+1)
    #print ticks
    plt.xticks(ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    #plt.show()
    plt.savefig(filename)
    
def plot_kvalues_dual(l1, l2, xlabel, ylabel, labels, filename, loc = 4, mark = "low"):
    '''
    Plotting the likelihoods given two dictionaries
    '''
    
    plt.figure()
    markers_on = [1,2]
    plt.plot(l1.keys(), l1.values(), label=labels[0], marker='o', color = "b")
    #mark value
    markers_first_x = []
    markers_first_y = []
    for k,v in l1.iteritems():
        if mark == "high":
            if v == max(l1.values()):
                markers_first_x.append(k)
                markers_first_y.append(v)
        if mark == "low":
            if v == min(l1.values()):
                markers_first_x.append(k)
                markers_first_y.append(v)
    plt.plot(markers_first_x, markers_first_y, 'rD')
    
    plt.plot(l2.keys(), l2.values(), label=labels[1], marker='o', color = "g") 
    #mark lowest value
    markers_second_x = []
    markers_second_y = []
    for k,v in l2.iteritems():
        if mark == "high":
            if v == max(l2.values()):
                markers_second_x.append(k)
                markers_second_y.append(v)
        if mark == "low":
            if v == min(l2.values()):
                markers_second_x.append(k)
                markers_second_y.append(v)
    plt.plot(markers_second_x, markers_second_y, 'rD')
     
    ticks = np.arange(min(l1.keys()), max(l1.keys())+1)
    print ticks
    plt.xticks(ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)   
    plt.legend(loc=loc) 
    #plt.show()
    plt.savefig(filename)
    
def plot_kvalues_trip(l1, l2, l3, xlabel, ylabel, labels, filename, loc = 4, mark = "low"):
    '''
    Plotting the likelihoods given three dictionaries
    '''
    plt.figure()

    plt.plot(l1.keys(), l1.values(), label=labels[0], marker='o', color='b')
    #mark value
    markers_first_x = []
    markers_first_y = []
    for k,v in l1.iteritems():
        if mark == "high":
            if v == max(l1.values()):
                markers_first_x.append(k)
                markers_first_y.append(v)
        if mark == "low":
            if v == min(l1.values()):
                markers_first_x.append(k)
                markers_first_y.append(v)
    plt.plot(markers_first_x, markers_first_y, 'rD')
    
    plt.plot(l2.keys(), l2.values(), label=labels[1], marker='o', color='r') 
    #mark lowest value
    markers_second_x = []
    markers_second_y = []
    for k,v in l2.iteritems():
        if mark == "high":
            if v == max(l2.values()):
                markers_second_x.append(k)
                markers_second_y.append(v)
        if mark == "low":
            if v == min(l2.values()):
                markers_second_x.append(k)
                markers_second_y.append(v)
    plt.plot(markers_second_x, markers_second_y, 'rD')
    
    plt.plot(l3.keys(), l3.values(), label=labels[2], marker='o', color='g') 
    #mark lowest value
    markers_third_x = []
    markers_third_y = []
    for k,v in l3.iteritems():
        if mark == "high":
            if v == max(l3.values()):
                markers_third_x.append(k)
                markers_third_y.append(v)
        if mark == "low":
            if v == min(l3.values()):
                markers_third_x.append(k)
                markers_third_y.append(v)
    plt.plot(markers_third_x, markers_third_y, 'rD')
    

     
    ticks = np.arange(min(l1.keys()), max(l1.keys())+1)
    print ticks
    plt.xticks(ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)   
    plt.legend(loc=loc) 
    #plt.show()

    plt.savefig(filename)
    
def plot_kvalues_quat(l1, l2, l3, l4, xlabel, ylabel, labels, filename, loc = 4, mark = "low"):
    '''
    Plotting the likelihoods given four dictionaries
    '''
    plt.figure()

    plt.plot(l1.keys(), l1.values(), label=labels[0], marker='o', color='b')
    #mark value
    markers_first_x = []
    markers_first_y = []
    for k,v in l1.iteritems():
        if mark == "high":
            if v == max(l1.values()):
                markers_first_x.append(k)
                markers_first_y.append(v)
        if mark == "low":
            if v == min(l1.values()):
                markers_first_x.append(k)
                markers_first_y.append(v)
    plt.plot(markers_first_x, markers_first_y, 'rD')
    
    plt.plot(l2.keys(), l2.values(), "--", label=labels[1], marker='o', color='b') 
    #mark lowest value
    markers_second_x = []
    markers_second_y = []
    for k,v in l2.iteritems():
        if mark == "high":
            if v == max(l2.values()):
                markers_second_x.append(k)
                markers_second_y.append(v)
        if mark == "low":
            if v == min(l2.values()):
                markers_second_x.append(k)
                markers_second_y.append(v)
    plt.plot(markers_second_x, markers_second_y, 'rD')
    
    plt.plot(l3.keys(), l3.values(), label=labels[2], marker='o', color='g') 
    #mark lowest value
    markers_third_x = []
    markers_third_y = []
    for k,v in l3.iteritems():
        if mark == "high":
            if v == max(l3.values()):
                markers_third_x.append(k)
                markers_third_y.append(v)
        if mark == "low":
            if v == min(l3.values()):
                markers_third_x.append(k)
                markers_third_y.append(v)
    plt.plot(markers_third_x, markers_third_y, 'rD')
    
    plt.plot(l4.keys(), l4.values(), "--", label=labels[3], marker='o', color='g') 
    #mark lowest value
    markers_fourth_x = []
    markers_fourth_y = []
    for k,v in l4.iteritems():
        if mark == "high":
            if v == max(l4.values()):
                markers_fourth_x.append(k)
                markers_fourth_y.append(v)
        if mark == "low":
            if v == min(l4.values()):
                markers_fourth_x.append(k)
                markers_fourth_y.append(v)
    plt.plot(markers_fourth_x, markers_fourth_y, 'rD')
     
    ticks = np.arange(min(l1.keys()), max(l1.keys())+1)
    print ticks
    plt.xticks(ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)   
    plt.legend(loc=loc) 
    #plt.show()

    plt.savefig(filename)

def plot_kvalues_six(l1, l2, l3, l4, l5, l6, xlabel, ylabel, labels, filename, loc = 4, mark = "low"):
    '''
    Plotting the likelihoods given four dictionaries
    '''
    plt.figure()

    plt.plot(l1.keys(), l1.values(), label=labels[0], marker='o', color='b')
    #mark value
    markers_first_x = []
    markers_first_y = []
    for k,v in l1.iteritems():
        if mark == "high":
            if v == max(l1.values()):
                markers_first_x.append(k)
                markers_first_y.append(v)
        if mark == "low":
            if v == min(l1.values()):
                markers_first_x.append(k)
                markers_first_y.append(v)
    plt.plot(markers_first_x, markers_first_y, 'rD')
    
    plt.plot(l2.keys(), l2.values(), "--", label=labels[1], marker='o', color='b') 
    #mark lowest value
    markers_second_x = []
    markers_second_y = []
    for k,v in l2.iteritems():
        if mark == "high":
            if v == max(l2.values()):
                markers_second_x.append(k)
                markers_second_y.append(v)
        if mark == "low":
            if v == min(l2.values()):
                markers_second_x.append(k)
                markers_second_y.append(v)
    plt.plot(markers_second_x, markers_second_y, 'rD')
    
    plt.plot(l3.keys(), l3.values(), label=labels[2], marker='o', color='r') 
    #mark lowest value
    markers_third_x = []
    markers_third_y = []
    for k,v in l3.iteritems():
        if mark == "high":
            if v == max(l3.values()):
                markers_third_x.append(k)
                markers_third_y.append(v)
        if mark == "low":
            if v == min(l3.values()):
                markers_third_x.append(k)
                markers_third_y.append(v)
    plt.plot(markers_third_x, markers_third_y, 'rD')
    
    plt.plot(l4.keys(), l4.values(), "--", label=labels[3], marker='o', color='r') 
    #mark lowest value
    markers_fourth_x = []
    markers_fourth_y = []
    for k,v in l4.iteritems():
        if mark == "high":
            if v == max(l4.values()):
                markers_fourth_x.append(k)
                markers_fourth_y.append(v)
        if mark == "low":
            if v == min(l4.values()):
                markers_fourth_x.append(k)
                markers_fourth_y.append(v)
    plt.plot(markers_fourth_x, markers_fourth_y, 'rD')
    
    plt.plot(l5.keys(), l5.values(), label=labels[4], marker='o', color='g') 
    #mark lowest value
    markers_third_x = []
    markers_third_y = []
    for k,v in l5.iteritems():
        if mark == "high":
            if v == max(l5.values()):
                markers_third_x.append(k)
                markers_third_y.append(v)
        if mark == "low":
            if v == min(l5.values()):
                markers_third_x.append(k)
                markers_third_y.append(v)
    plt.plot(markers_third_x, markers_third_y, 'rD')
    
    plt.plot(l6.keys(), l6.values(), "--", label=labels[5], marker='o', color='g') 
    #mark lowest value
    markers_fourth_x = []
    markers_fourth_y = []
    for k,v in l6.iteritems():
        if mark == "high":
            if v == max(l6.values()):
                markers_fourth_x.append(k)
                markers_fourth_y.append(v)
        if mark == "low":
            if v == min(l6.values()):
                markers_fourth_x.append(k)
                markers_fourth_y.append(v)
    plt.plot(markers_fourth_x, markers_fourth_y, 'rD')
    
    ticks = np.arange(min(l1.keys()), max(l1.keys())+1)
    print ticks
    plt.xticks(ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)   
    plt.legend(loc=loc) 
    #plt.show()

    plt.savefig(filename)

def plot_kvalues_barchart_dual(l1, l2, xlabel, ylabel, labels, filename, loc = 4):
    '''
    Plotting the likelihoods given two dictionaries barchart
    '''
    
    ind = np.arange(len(l1))
    width = 0.35
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    rects1 = ax.bar(ind, l1.values(), width, color='b')
    rects2 = ax.bar(ind+width, l2.values(), width, color='g')
    
    
    #plt.bar(l1.keys(), l1.values(), label=labels[0])
    #plt.bar(l2.keys(), l2.values(), label=labels[1]) 
     
    #ticks = np.arange(min(l1.keys()), max(l1.keys())+1)
    #print ticks
    ax.set_xticks(ind+width)
    ax.set_xticklabels( tuple([str(x) for x in l1.keys()]) )
    
    ax.legend( (rects1[0], rects2[0]), tuple([str(x) for x in labels]), loc = loc )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)   
    #plt.legend(loc=4) 
    #plt.show()
    plt.savefig(filename)

def plot_kvalues_barchart_quat(l1, l2, l3, l4, xlabel, ylabel, labels, filename, loc = 4):
    '''
    Plotting the likelihoods given two dictionaries barchart
    '''
    
    ind = np.arange(len(l1))
    width = 0.35
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    rects1 = ax.bar(ind, l1.values(),  width=width, color='b')
    rects2 = ax.bar(ind+width, l2.values(),hatch="//", width=width, color='b')
    
    rects3 = ax.bar(ind, l3.values(), width=width, color='g')
    rects4 = ax.bar(ind+width, l4.values(),hatch="//", width=width,  color='g')
    
    #plt.bar(l1.keys(), l1.values(), label=labels[0])
    #plt.bar(l2.keys(), l2.values(), label=labels[1]) 
     
    #ticks = np.arange(min(l1.keys()), max(l1.keys())+1)
    #print ticks
    ax.set_xticks(ind+width)
    ax.set_xticklabels( tuple([str(x) for x in l1.keys()]) )
    
    ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), tuple([str(x) for x in labels]), loc = loc )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)   
    #plt.legend(loc=4) 
    #plt.show()
    plt.savefig(filename)
    
def plot_kvalues_barchart_six(l1, l2, l3, l4, l5, l6, xlabel, ylabel, labels, filename, loc = 4):
    '''
    Plotting the likelihoods given two dictionaries barchart
    '''
    
    ind = np.arange(len(l1))
    width = 0.25
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    rects1 = ax.bar(ind, l1.values(),  width=width, color='b')
    rects2 = ax.bar(ind+0.25, l2.values(),hatch="//", width=width, color='b')
    
    rects3 = ax.bar(ind+0.50, l3.values(), width=width, color='r')
    rects4 = ax.bar(ind+0.75, l4.values(),hatch="//", width=width,  color='r')
    
    rects5 = ax.bar(ind+0.25, l5.values(), width=width, color='g')
    rects6 = ax.bar(ind+0.50, l6.values(),hatch="//", width=width,  color='g')
    
    #plt.bar(l1.keys(), l1.values(), label=labels[0])
    #plt.bar(l2.keys(), l2.values(), label=labels[1]) 
     
    #ticks = np.arange(min(l1.keys()), max(l1.keys())+1)
    #print ticks
    ax.set_xticks(ind+width*2)
    ax.set_xticklabels( tuple([str(x) for x in l1.keys()]) )
    
    ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]), tuple([str(x) for x in labels]), loc = loc )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)   
    #plt.legend(loc=4) 
    #plt.show()
    plt.savefig(filename)
    
def plot_distr(l, xlabel, ylabel, filename):
    '''
    Plotting a given distribution
    '''
    plt.figure()
    plt.plot(l)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    #plt.show()
    plt.savefig(filename)

def dict_table(d, header, caption, label, filename, horicontal = False, columns = 2):
    '''
    Provides latex table syntax given a dictionary
    '''
    string = "\\begin{table}[ht!]"
    string += "\n"
    string += "\\centering"
    string += "\n"
    string += "\\caption{" + caption + "}"
    string += "\n"
    
    if horicontal:
        string += "\\begin{tabular}{|" + (len(d.keys()) + 1)*"l|" + "} \\hline"
        string += "\n"
        string += header[0] + " & " + " & ".join([str(x) for x in d.keys()]) + "\\\\ \\hline"
        string += "\n"
        string += header[1] + " & " + " & ".join([str(x) for x in d.values()]) + "\\\\ \\hline"
        string += "\n"
    else:
        string += "\\begin{tabular}{|" + columns*"l|" + "} \\hline"
        string += "\n"
        string += " & ".join(header) + " \\\\ \\hline \\hline"
        string += "\n"
        
        for k, v in d.iteritems():
            if isinstance(v, types.ListType):
                string += str(k) + " & " + " & ".join([str(x) for x in v]) + " \\\\ \\hline"
            else:
                string += str(k) + " & " + str(v) + "\\\\ \\hline"
            string += "\n"
    
    string += "\\end{tabular}"
    string += "\n"
    string += "\\label{" + label + "}"
    string += "\n"
    string += "\\end{table}"
    
    print string
    
    with open(filename, "w") as text_file:
        text_file.write(string)
    
def transition_dicts_to_matrix(transition_dict_dict):
        i_indices = []
        j_indices = []
        values = []
        #vocabulary = self.vocabulary_

        vocab = set()

        for transition_dict_key in transition_dict_dict.keys():
            #print "key", transition_dict_dict[cooc_dict_key]
            transition_dict = transition_dict_dict[transition_dict_key]
            for term, count in transition_dict.iteritems():
                vocab.add(term)
        
        vocabulary = dict()
        i = 0
        for x in sorted(vocab):
            vocabulary[x] = i
            i += 1
             
        print vocabulary   

        #print transition_dict_dict
        counter = 0
        for transition_dict_key in transition_dict_dict.keys():
            #print "key", transition_dict_dict[cooc_dict_key]
            transition_dict = transition_dict_dict[transition_dict_key]
            for term, count in transition_dict.iteritems():
                i = vocabulary.get(transition_dict_key[0])
                j = vocabulary.get(term)
                counter += count
                #if transition_dict_key == 0 and term == 9089624:
                    #print i, j, count, "bla"
                #print "j", j
                if i is not None and j is not None: 
                    #if i == 0 and j == 277457:
                        #print i, j, count
                    i_indices.append(i)                   
                    j_indices.append(j)
                    values.append(count)
            # free memory as we go
            transition_dict.clear()

        print len(i_indices)
        print len(j_indices)
        print len(values)

        shape = (max(vocabulary.itervalues()) + 1, max(vocabulary.itervalues()) + 1)
        print shape
        #print len(values)
        spmatrix = sp.coo_matrix((values, (i_indices, j_indices)),
                                 shape=shape, dtype=np.dtype(float))
        
        return sp.csr_matrix(spmatrix), counter, vocabulary

        

