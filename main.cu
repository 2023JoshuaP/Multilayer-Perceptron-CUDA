#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <random>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "include/Matrix.cuh"
#include "include/Activation.cuh"
#include "include/MultiLayerPerceptron.cuh"
#include "include/DataLoader.cuh"

using namespace std;

int main(int argc, char* argv[]) {
    string symbols_path = "HASYv2/symbols.csv";
    
    if (argc >= 2) {
        symbols_path = argv[1];
    }
    DataLoader::load_symbols(symbols_path);

    double total_accuracy = 0.0;
    int num_folds = 1;

    for (int i = 1; i <= num_folds; ++i) {
        cout << "\n=== Fold " << i << " ===" << endl;
        string base_dir = "HASYv2/classification-task/fold-" + to_string(i) + "/";
        string train_csv = base_dir + "train.csv";
        string test_csv = base_dir + "test.csv";

        cout << "Loading training data..." << endl;
        auto [X_train, y_train] = DataLoader::load_csv_data(train_csv, base_dir);
        cout << "Loading test data..." << endl;
        auto [X_test, y_test] = DataLoader::load_csv_data(test_csv, base_dir);
        
        auto activation = make_shared<ReLU>();
        int classes = DataLoader::NUM_CLASSES;
        MultiLayerPerceptron mlp({1024, 512, 256, 128, classes}, activation, 0.001, 0.9, 1e-5, 42);

        constexpr int EPOCHS = 100;
        constexpr int BATCH_SIZE = 128;
        mlp.train(X_train, y_train, EPOCHS, BATCH_SIZE, &X_test, &y_test, true, 80);

        Matrix test_predictions = mlp.predict(X_test);
        double test_accuracy = DataLoader::accuracy(test_predictions, y_test);
        cout << "Fold " << i << " Test Accuracy: " << fixed << setprecision(2) << test_accuracy << "%" << endl;
        total_accuracy += test_accuracy;

        int test_n = X_test.rows;
        int num_cols = test_predictions.cols;
        constexpr int NUM_SAMPLES = 20;
        constexpr int IMG_SIZE = 32;
        int flat_size = IMG_SIZE * IMG_SIZE;

        vector<double> h_pred(test_n * num_cols);
        vector<double> h_true(test_n * num_cols);
        test_predictions.toHost(h_pred.data());
        y_test.toHost(h_true.data());

        vector<double> h_X(test_n * flat_size);
        X_test.toHost(h_X.data());

        vector<int> indices(test_n);
        iota(indices.begin(), indices.end(), 0);
        mt19937 rng_vis(123);
        shuffle(indices.begin(), indices.end(), rng_vis);
        int n_show = min(NUM_SAMPLES, test_n);

        cout << "\n=== Sample Predictions (Test Set) ===" << endl;
        cout << left << setw(8) << "Sample"
             << setw(20) << "Predicted"
             << setw(20) << "True"
             << "Result" << endl;
        cout << string(55, '-') << endl;

        vector<int> pred_labels(n_show), true_labels(n_show);
        for (int s = 0; s < n_show; s++) {
            int idx = indices[s];
            int pl = 0; double best = h_pred[idx * num_cols];
            for (int j = 1; j < num_cols; j++) {
                if (h_pred[idx * num_cols + j] > best) {
                    best = h_pred[idx * num_cols + j];
                    pl = j;
                }
            }
            pred_labels[s] = pl;

            int tl = 0;
            for (int j = 1; j < num_cols; j++) {
                if (h_true[idx * num_cols + j] > h_true[idx * num_cols + tl]) {
                    tl = j;
                }
            }
            true_labels[s] = tl;

            bool correct = (pl == tl);
            cout << left << setw(8) << (s + 1)
                 << setw(20) << DataLoader::CLASS_NAMES[pl]
                 << setw(20) << DataLoader::CLASS_NAMES[tl]
                 << (correct ? "OK" : "FAIL") << endl;
        }

        constexpr int SCALE = 3;
        constexpr int GRID_COLS = 5;
        constexpr int GRID_ROWS = 4;
        int cell_w = IMG_SIZE * SCALE;
        int cell_h = IMG_SIZE * SCALE + 40;
        int grid_n = min(n_show, GRID_COLS * GRID_ROWS);

        cv::Mat grid(GRID_ROWS * cell_h, GRID_COLS * cell_w, CV_8UC3, cv::Scalar(40, 40, 40));

        for (int s = 0; s < grid_n; s++) {
            int idx = indices[s];
            int row = s / GRID_COLS;
            int col = s % GRID_COLS;

            cv::Mat img(IMG_SIZE, IMG_SIZE, CV_8UC1);
            for (int r = 0; r < IMG_SIZE; r++) {
                for (int c = 0; c < IMG_SIZE; c++) {
                    img.at<uint8_t>(r, c) = static_cast<uint8_t>(
                        h_X[idx * flat_size + r * IMG_SIZE + c] * 255.0);
                }
            }

            cv::Mat img_scaled, img_color;
            cv::resize(img, img_scaled, cv::Size(cell_w, IMG_SIZE * SCALE), 0, 0, cv::INTER_NEAREST);
            cv::cvtColor(img_scaled, img_color, cv::COLOR_GRAY2BGR);

            bool correct = (pred_labels[s] == true_labels[s]);
            cv::Scalar border_color = correct ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 220);

            cv::rectangle(img_color, cv::Point(0, 0),
                          cv::Point(cell_w - 1, IMG_SIZE * SCALE - 1), border_color, 2);

            int x0 = col * cell_w;
            int y0 = row * cell_h;
            img_color.copyTo(grid(cv::Rect(x0, y0, cell_w, IMG_SIZE * SCALE)));
            string pred_text = "P: " + DataLoader::CLASS_NAMES[pred_labels[s]];
            string true_text = "T: " + DataLoader::CLASS_NAMES[true_labels[s]];

            cv::putText(grid, pred_text, cv::Point(x0 + 4, y0 + IMG_SIZE * SCALE + 14),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, border_color, 1);
            cv::putText(grid, true_text, cv::Point(x0 + 4, y0 + IMG_SIZE * SCALE + 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1);
        }

        cv::putText(grid, "MLP CUDA - Test Predictions (Fold " + to_string(i) + ")",
                    cv::Point(10, GRID_ROWS * cell_h - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 180, 180), 1);

        cv::imshow("MLP CUDA - Predictions", grid);
        cout << "\nPress any key on the image window to continue..." << endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    cout << "\nAverage Test Accuracy over " << num_folds << " folds: "
         << fixed << setprecision(2) << (total_accuracy / num_folds) << "%" << endl;

    return 0;
}