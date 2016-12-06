from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt


ce_train = list()
frac_train = list()
ce_val = list()
frac_val = list()
plot = True
small = False
test = True

def run_logistic_regression(hyperparameters):
    # TODO specify training data
    if small:
        train_inputs, train_targets = load_train_small()
    else:
        train_inputs, train_targets = load_train()

    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = 0.01 * np.random.randn(M+1, 1)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]
        if plot:
            ce_train.append(cross_entropy_train)
            ce_val.append(cross_entropy_valid)
            frac_train.append(frac_correct_train)
            frac_val.append(frac_correct_valid)
    if test:
        predictions_test = logistic_predict(weights, test_inputs)
        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)
        # print some stats
        print ("TEST CE:{:.6f} "
               "TEST FRAC:{:2.2f}").format(
            cross_entropy_test, frac_correct_test * 100)
    return logging, cross_entropy_train, cross_entropy_valid, frac_correct_train, frac_correct_valid

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.06,
                    'weight_regularization': True, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 1000,
                    'weight_decay': 1.0 # related to standard deviation of weight prior
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging_run, _, _, _, _ = run_logistic_regression(hyperparameters)
        logging += logging_run
    logging /= num_runs

    # #For 2.3
    # w_alphas = [1.0, 0.1, 0.01, 0.001]
    # smalls = [True, False]
    # for plot_small in smalls:
    #     cross_entropy_train_set = list()
    #     frac_correct_train_set = list()
    #     cross_entropy_val_set = list()
    #     frac_correct_val_set = list()
    #     for w_alpha in w_alphas:
    #
    #         hyperparameters['weight_decay'] = w_alpha
    #         _, train_ce, val_ce, train_frac, val_frac = run_logistic_regression(hyperparameters)
    #         cross_entropy_train_set.append(train_ce)
    #         frac_correct_train_set.append(train_frac)
    #         cross_entropy_val_set.append(val_ce)
    #         frac_correct_val_set.append(val_frac)
    #
    #     line1, = plt.semilogx(w_alphas, cross_entropy_train_set, label ="Training Cross Entropy")
    #     line2, = plt.semilogx(w_alphas, cross_entropy_val_set, label ="Validation Cross Entropy")
    #     plt.legend(handles = [line1, line2], loc = 2)
    #     plt_title = "Cross Entropy Vs Alpha "
    #     if plot_small:
    #         plt.title(plt_title + "(Small Training Set)")
    #     else:
    #         plt.title(plt_title + "(Large Training Set)")
    #     plt.xlabel("Alpha")
    #     plt.ylabel("Cross Entropy")
    #     plt.grid(True)
    #     plt.show()
    #
    #     line1, = plt.semilogx(w_alphas, frac_correct_train_set, label="Training Cross Entropy")
    #     line2, = plt.semilogx(w_alphas, frac_correct_val_set, label="Validation Cross Entropy")
    #     plt.legend(handles=[line1, line2], loc=2)
    #     plt_title = "Classification Rate Vs Alpha "
    #     if plot_small:
    #         plt.title(plt_title + "(Small Training Set)")
    #     else:
    #         plt.title(plt_title + "(Large Training Set)")
    #     plt.xlabel("Alpha")
    #     plt.ylim([0.8, 1.2])
    #     plt.ylabel("Classification Rate")
    #     plt.grid(True)
    #     plt.show()


    #2.2 and 2.3
    if plot:
        train_line,  = plt.plot(ce_train, label ="Training Cross Entropy")
        val_line,  = plt.plot(ce_val, label = "Validation Cross Entropy")
        plt.legend(handles=[train_line, val_line])
        title = "Cross Entropy Vs. Iteration "
        if small:
            plt.title(title + "(Small Set)")
        else:
            plt.title(title + "(Large Set)")
        plt.xlabel("Iteration")
        plt.ylabel("Cross Entropy")
        plt.grid(True)
        plt.show()

    # TODO generate plots
