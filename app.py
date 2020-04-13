###
__author__: "Pooja Kamble"
__version__: "1.1"

# Program implements the 1D classification algorithm for given training data.

import csv
import math
import matplotlib.pyplot as plt


# The method returns the best threshold, best_cost_fun, cost_fun_plot, ROC.
def threshold_setting_1D_algorithm(bins):
    best_cost_fun = math.inf
    best_threshold = 0
    best_ind = 0
    cost_fun_plot = {}
    ind = 0
    ROC = {}
    non_reckless = 0
    for threshold in bins.keys():
        ROC[threshold] = {'TPR': 0, 'FAR':0, 'CF':0}
        TN = 0
        TP = 0
        FN = 0
        FP = 0

        for keys, val in bins.items():
            if keys <= threshold:
                FN += len(val[1])
                TN += len(val[0])
            else:
                TP += len(val[1])
                FP += len(val[0])
        # Uncomment this to have the minimize cost function to 
        # penalize more for missed speeders
        # cost_fun = calculateCost(FN, FP, True)

        #normal cost function
        cost_fun = calculateCost(FN, FP)

        cost_fun_plot[threshold] = cost_fun
        if(cost_fun <= best_cost_fun):
            best_cost_fun = cost_fun
            best_threshold = threshold
            best_ind = ind
        ROC[threshold]['TPR'] = TP
        ROC[threshold]['FAR'] = FP
        ROC[threshold]['CF'] = cost_fun
        non_reckless += FN
        ind += 1
    return best_threshold, best_cost_fun, cost_fun_plot, ROC

# returns the number of aggressive drivers let through.
def calculate_aggr_drivers(best_threshold, bins):
    num_aggr_drivers = 0
    for keys,values in bins.items():
        if keys <= best_threshold:
            num_aggr_drivers += len(values[1])
    return num_aggr_drivers

# returns the number of non reckless drivers pulled over.
def calculate_nonRecklessDrivers(best_threshold, bins):
    non_reckless_drivers= 0
    for keys,values in bins.items():
        if keys > best_threshold:
            non_reckless_drivers += len(values[0])
    return non_reckless_drivers

# generic cost function can be used in two ways.
def calculateCost(FN, FP, flag = False):
    if(flag):
        return 2*FN + FP
    else:
        return FN + FP


# main function for processing input files and calling different functions to classfiy data.
def main():
    # System file path for input file
    data_path = ""
    bin_size = 0.5
    f= data_path + "DATA_v2191_FOR_CLASSIFICATION_using_Threshold.csv"

    bins = {}
    
    # binning
    for i in range(45, 80):
        bin_val = float(i)
        bins[bin_val] = {0:[],1:[]}
        bins[bin_val+ 0.5] ={0:[],1:[]}

    bins[80.0] = {0:[],1:[]}
    with open(f) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        tuple = {0:[],1:[]}
        for row in reader:
            speed = float(row[0])
            aggr = int(row[1])
            binkey = (round(speed/bin_size))*bin_size

            if binkey in bins.keys():
                bins[binkey][aggr].append(speed)
    

    best_speed, best_cost_fun, cost_fun_plot, ROC = threshold_setting_1D_algorithm(bins)
    print("Best threshold: ", best_speed, "mph")
    print("Aggressive drivers: ",calculate_aggr_drivers(best_speed, bins))
    print("non_reckless_drivers: ",calculate_nonRecklessDrivers(best_speed, bins))
    plotgraph(cost_fun_plot)
    plotgraph1(ROC, best_speed, best_cost_fun)


# Plots the graph for Cost fun vs threshold.
def plotgraph(cost_fun_plot):
    plt.figure()
    plt.title("Cost function as a function of threshold")
    plt.xlabel('Threshold')
    plt.ylabel('Cost function')
    x_axis = []
    y_axis = []

    for key, val in cost_fun_plot.items():
        x_axis.append(key)
        y_axis.append(val)
    plt.plot(x_axis,y_axis, '.-')
    plt.show()

# plots the graph of ROC
def plotgraph1(ROC, best_speed, best_cost):
    plt.figure()
    plt.title("ROC curve by Threshold", y=-0.01)
    plt.xlabel('False Alarm Rate(FAR)')
    plt.ylabel('True Positive Rate(TPR)')
    x_axis = []
    y_axis = []

    annote_x = 0.0
    annpte_y = 0.0
    cf_x = 0.0
    cf_y = 0.0
    for key, val in ROC.items():
        x = val['FAR']
        y = val['TPR']
        cf = val['CF']
        if cf == best_cost:
            cf_x = x
            cf_y = y
        x_axis.append(x)
        y_axis.append(y)
        if key == best_speed:
            annote_x = x
            annote_y = y
        if(cf_y >0):
            plt.plot(cf_x, cf_y, "--ro")
    plt.plot(x_axis,y_axis, '.-')
    plt.tick_params(labeltop=True, labelright=True, rotation=90)
    plt.tick_params(top= True, right= True)
    plt.annotate('Threshold: ' + str(best_speed)+ "mph | FAR: " + str(annote_x)+", TPR: "+ str(annote_y), xy=(annote_x,annote_y),
                 xytext=(annote_x*0.8 ,annote_y*0.8),
                 arrowprops=dict(facecolor='black', shrink=0.10))
    plt.show()

# main
if __name__ == '__main__':
    main()