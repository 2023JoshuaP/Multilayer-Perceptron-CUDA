#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include "include/Matrix.cuh"
#include "include/Activation.cuh"
#include "include/MultiLayerPerceptron.cuh"
#include "include/DataLoader.cuh"

using namespace std;

int main(int argc, char* argv[]) {
    string symbols_path = "HASYv2/symbols.csv";
    string train_csv = "HASYv2/classification-task/fold-1/train.csv";
    string test_csv = "HASYv2/classification-task/fold-1/test.csv";
    string base_dir = "HASYv2/classification-task/fold-1/";

    if (argc == 4) {
        symbols_path = argv[1];
        train_csv = argv[2];
        test_csv = argv[3];
    }

    DataLoader::load_symbols(symbols_path);
    cout << "Loading training data..." << endl;
    auto [X_train, y_train] = DataLoader::load_csv_data(train_csv, base_dir);
    cout << "Loading test data..." << endl;
    auto [X_test, y_test] = DataLoader::load_csv_data(test_csv, base_dir);
    
    auto activation = make_shared<ReLU>();
    int classes = DataLoader::NUM_CLASSES;
    MultiLayerPerceptron mlp({1024, 512, 256, 128, classes}, activation, 0.001, 0.9, 1e-5, 42);

    constexpr int EPOCHS = 200;
    constexpr int BATCH_SIZE = 128;
    auto history = mlp.train(X_train, y_train, EPOCHS, BATCH_SIZE, &X_test, &y_test, true, 80);

    cout << "\nEvaluating on train set..." << endl;
    Matrix train_predictions = mlp.predict(X_train);
    double train_accuracy = DataLoader::accuracy(train_predictions, y_train);
    cout << "Final Training Accuracy: " << fixed << setprecision(2) << train_accuracy << "%" << endl;

    cout << "\nEvaluating on test set..." << endl;
    Matrix test_predictions = mlp.predict(X_test);
    double test_accuracy = DataLoader::accuracy(test_predictions, y_test);
    cout << "Final Test Accuracy: " << fixed << setprecision(2) << test_accuracy << "%" << endl;

    return 0;
}