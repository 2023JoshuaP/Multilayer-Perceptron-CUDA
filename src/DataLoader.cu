#include "DataLoader.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <utility>
#include <map>
#include <vector>
#include <string>

using namespace std;
namespace fs = std::filesystem;

vector<string> DataLoader::CLASS_NAMES = {};
map<int, int> DataLoader::SYMBOL_ID_TO_INDEX = {};
int DataLoader::NUM_CLASSES = 0;

/* Load symbols from CSV file */
void DataLoader::load_symbols(const string &symbols_csv) {
    ifstream file(symbols_csv);
    if (!file.is_open()) {
        cerr << "Error: Could not open symbols CSV." << endl;
        return;
    }

    CLASS_NAMES.clear();
    SYMBOL_ID_TO_INDEX.clear();

    string line;
    getline(file, line);

    int index = 0;
    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        stringstream ss(line);
        string symbol_id, latex_name;
        getline(ss, symbol_id, ',');
        getline(ss, latex_name, ',');

        int symbol_int_id = stoi(symbol_id);
        SYMBOL_ID_TO_INDEX[symbol_int_id] = index;
        CLASS_NAMES.push_back(latex_name);
        index++;
    }

    NUM_CLASSES = index;
    file.close();
    cout << "Loaded" << NUM_CLASSES << "symbols "
}

/* Load data from CSV file */
pair<Matrix, Matrix> DataLoader::load_csv_data(const string &csv_path, const string &base_dir, int img_size) {
    if (NUM_CLASSES == 0) {
        cerr << "Could not load data, symbols not loaded." << endl;
        throw runtime_error("Symbols not loaded");
    }

    int flat_size = img_size * img_size;
    ifstream file(csv_path);
    if (!file.is_open()) {
        cerr << "Could not open csv file: " << csv_path << endl;
        throw runtime_error("Could not open csv file");
    }

    vector<vector<double>> rows_x;
    vector<int> labels;
    string line;
    getline(file, line);

    int loaded = 0, skipped = 0;
    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        stringstream ss(line);
        string img_rel_path, symbol_id_str;
        getline(ss, img_rel_path, ',');
        getline(ss, symbol_id_str, ',');

        int symbol_id = stoi(symbol_id_str);
        auto it = SYMBOL_ID_TO_INDEX.find(symbol_id);
        if (it == SYMBOL_ID_TO_INDEX.end()) {
            skipped++;
            continue;
        }
        int label = it->second;

        fs::path full_path = fs::path(base_dir) / img_rel_path:
        cv::Mat image = cv::imread(full_path.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            skipped++;
            cerr << "Skipping ";
            continue;
        }

        if (image.rows != img_size || image.cols != img_size) {
            cv::resize(image, image, cv::Size(img_size, img_size), 0, 0, cv::INTER_AREA);
        }

        vector<double> flat_image(flat_size);
        for (int i = 0; i < img_size; i++) {
            for (int j = 0; j < img_size; j++) {
                flat_image[i * img_size + j] = static_cast<double>(image.at<uint8_t>(i, j)) / 255.0;
            }
        }

        rows_x.push_back(flat_image);
        labels.push_back(label);
        loaded++;

        if (loaded % 5000 == 0) {
            cout << "Loaded " << loaded << " images..." << endl;
        }
    }

    int N = static_cast<int>(rows_x.size());
    if (N == 0) {
        cerr << "No valid images found" << endl;
        throw runtime_error("No valid images found");
    }

    cout << "Loaded " << loaded << " images" << endl;
    if (skipped > 0) {
        cout << "Skipped " << skipped << " images" << endl;
    }

    vector<double> h_X(N * flat_size);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < flat_size; j++) {
            h_X[i * flat_size + j] = rows_x[i][j];
        }
    }

    Matrix X(N, flat_size);
    X.from_host(h_X.data());

    vector<double> h_Y(N * NUM_CLASSES, 0.0);
    for (int i = 0; i < N; i++) {
        h_Y[i * NUM_CLASSES + labels[i]] = 1.0;
    }

    Matrix Y(N, NUM_CLASSES);
    Y.from_host(h_Y.data());

    return {move(X), move(Y)};
}