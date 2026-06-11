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