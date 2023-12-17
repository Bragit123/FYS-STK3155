# Figures
In this folder we save all the figures created in our code. The figures starting
with *cnn* are the results from the convolutional neural network, such that
those starting with *cnn_test* are validation accuracies, and those starting
with *cnn_train* are training accuracies. They are then followed by the name of
the hidden activation function used, such that *cnn_test_acc_relu.pdf* is the
validation accuracy plot with the RELU activation function.

The figures starting with *val_accs_NN* are the resulting figures from the
regular neural network, with the end of the filename stating the name of the
hidden layer activation function. Thus, for instance, *val_accs_NN_RELU.pdf* is
the validation accuracy plot with the RELU activation function.

The last figures are *val_accs_adaboost.pdf*, *val_accs_gradboost.pdf* and
*val_accs_xgboost.pdf*, which are the resulting figures for the boosting methods
adaboost, gradient boost and XG boost respectively.