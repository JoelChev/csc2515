import numpy as np
from run_knn import run_knn
import matplotlib.pyplot as plt
import utils as utils

small = False

if __name__ == "__main__":
    k_values = [1,3,5,7,9]
    if small:
        train_inputs, train_targets = utils.load_train_small()
    else:
        train_inputs, train_targets = utils.load_train()
    valid_inputs, valid_targets = utils.load_valid()
    test_inputs, test_targets = utils.load_test()
    result_list = list()

    for k_val in k_values:
        knn_labels = \
            run_knn(k_val,train_inputs, train_targets, valid_inputs)
        valid_targets_array = valid_targets.astype('int64')
        knn_labels_array = knn_labels.astype('int64')
        correct = np.equal(valid_targets_array, knn_labels_array)
        total_correct = float(np.sum(correct))
        correct_val_percentage = 100 * total_correct/len(knn_labels)
        result_list.append(correct_val_percentage)
        print "k = %d, percent = %d, " % (k_val, correct_val_percentage)

    plt.plot(k_values, result_list)
    title = "Validation Classification Rate Vs. K "
    if small:
        plt.title(title + "(Small Set)")
    else:
        plt.title(title + "(Large Set)")
    plt.xlabel('Value of K')
    plt.ylabel('Correct Predictions (%) ')
    plt.ylim([50, 100])
    plt.grid(True)
    plt.show()

    if small:
        #I'm going to take k = 1,3,5 for the test performance
        test_k_values = k_values[0:3]
    else:
        #I'm going to take k = 3,5,7 for the test performance
        test_k_values = k_values[1:4]
    test_result_list = list()
    for k_val in test_k_values:
        knn_labels = \
            run_knn(k_val, train_inputs, train_targets, test_inputs)
        test_targets_array = test_targets.astype('int64')
        knn_labels_array = knn_labels.astype('int64')
        correct = np.equal(test_targets_array, knn_labels_array)
        total_correct = float(np.sum(correct))
        correct_percentage = 100 * total_correct / len(knn_labels)
        test_result_list.append(correct_percentage)
        print "k = %d, percent = %d, " % (k_val, correct_percentage)

    print len(test_result_list)
    plt.plot(test_k_values, test_result_list)
    title = 'Test Classification Rate Vs. K'
    if small:
        plt.title(title + "(Small Set)")
    else:
        plt.title(title + "(Large Set)")
    plt.xlabel('Value of K')
    plt.ylabel('Correct Predictions (%) ')
    plt.ylim([50, 100])
    plt.grid(True)
    plt.xticks(range(min(test_k_values), max(test_k_values)+1))
    plt.show()
