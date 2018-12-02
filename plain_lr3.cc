/*

经过了特征处理, l2 regularization做了之后，稍有改进 
# map 1: x  ->  [0, 100]  ->  [30, 100]
# map 2: x  ->  [0, 50]  ->  [15, 50]
# map 11:x  ->  [0, 30]  ->  [10, 30]

之后如论如何调整l2 reg的参数，都无法提高auc，说明此时lr模型出现欠拟合占主导

train auc:0.532509      test auc:0.531455
train auc:0.546925      test auc:0.545621
train auc:0.569203      test auc:0.56712
train auc:0.594232      test auc:0.591524
train auc:0.616049      test auc:0.612137
train auc:0.632369      test auc:0.627554
train auc:0.643715      test auc:0.638935
train auc:0.651367      test auc:0.646823
train auc:0.656372      test auc:0.65194
train auc:0.659752      test auc:0.655025
train auc:0.662218      test auc:0.656944
train auc:0.664149      test auc:0.658269
train auc:0.665644      test auc:0.659164
train auc:0.666849      test auc:0.659804
train auc:0.66788       test auc:0.66027
train auc:0.668789      test auc:0.660632
train auc:0.669609      test auc:0.660891
train auc:0.670336      test auc:0.661093
train auc:0.670996      test auc:0.661234
train auc:0.671614      test auc:0.661334
train auc:0.672188      test auc:0.66142
train auc:0.672724      test auc:0.661498
train auc:0.673231      test auc:0.661564
train auc:0.673717      test auc:0.661616
train auc:0.674179      test auc:0.661642
train auc:0.674626      test auc:0.661673
train auc:0.675054      test auc:0.661686
train auc:0.675464      test auc:0.661702
train auc:0.675859      test auc:0.661711
train auc:0.676245      test auc:0.661723
train auc:0.676617      test auc:0.661731
train auc:0.676981      test auc:0.661731
train auc:0.677334      test auc:0.661734
train auc:0.677676      test auc:0.66173
train auc:0.67801       test auc:0.661737
train auc:0.678334      test auc:0.661733
train auc:0.678648      test auc:0.66172
train auc:0.678957      test auc:0.661712
train auc:0.67926       test auc:0.661707
train auc:0.679553      test auc:0.661695
train auc:0.67984       test auc:0.661685
train auc:0.68012       test auc:0.661665
train auc:0.680397      test auc:0.661651

*/

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>

using namespace std;

typedef struct SFeat {
    int32_t index;
    float value;
} Feat;

typedef struct SSample {
    int32_t label;

    vector<Feat> feats;
} Sample;

typedef struct SPair {
    int32_t label;
    float y_pred;
} Pair;

typedef struct SModel {
    vector<float> ws;
    float bias;
} Model;

Model g_model;
vector<Sample> g_samples;
vector<Sample> g_test_samples;
int32_t g_vlen = 12;
int32_t g_app_num = 1000;

const int32_t NUMBER = 400;

int32_t g_used_vs[] = {1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0};
int32_t g_used_vlen = 4;

float g_learning_rate = 0.0005f;
float l1 = 0.0001f;
float l2 = 0.00005f;

string g_file_path = "mod_libsvm.data";

bool is_multiple(int32_t index, int32_t num) {
    return index % num == 0;
}

bool use_feat(int32_t index) {
  int32_t v = index  % g_vlen;
  return v == 1 || v == 2 || v == 11;
}

float sigmoid(float in) {
  return 1 / (1 + exp(-in));
}

void init_data() {
    string line;
    ifstream fs(g_file_path.c_str());

    int32_t P = 0, N = 0;
    while (getline(fs, line)) {
        size_t pos = line.find('\t');
        line = line.substr(pos + 1);
        Sample s;
        istringstream ls(line);
        string label_str;
        getline(ls, label_str, ' ');
        s.label = atoi(label_str.c_str());
        if (s.label == 1) {
            ++P;
        } else {
            ++N;
        }

        string feat_str;
        while (getline(ls, feat_str, ' ')) {
            size_t pos = feat_str.find(':');
            string index_str = feat_str.substr(0, pos);
            string val_str = feat_str.substr(pos + 1);

            Feat feat;
            feat.index = atoi(index_str.c_str());
            feat.value = atof(val_str.c_str()) * 0.01f;
            if (feat.value > 1.0f) {
              feat.value = 1.0f;
            }
            if (use_feat(feat.index)) {
              s.feats.push_back(feat);
            }
        }

        if (s.feats.empty()) {
          continue;
        }
        int rnum = rand() % 100;
        if (rnum < 20) {
          g_test_samples.push_back(s);
        } else {
          g_samples.push_back(s);
        }
    }

    fs.close();
    cout << "train samples:" << g_samples.size() << ", P:" << P << ", N:" << N << endl;
    cout << "test samples:" << g_test_samples.size() << ", P:" << P << ", N:" << N << endl;
}

void init_model() {
    g_model.bias = 0;

    g_model.ws.resize(g_app_num * 12, -0.01f);
}

bool comp(Pair a, Pair b) {
    return a.y_pred < b.y_pred;
}

void output_model() {
  cout << "bias:" << g_model.bias << "\n";
  cout << "ws:";
  for (int i=0; i<1000; ++i) {
    if (use_feat(i)) {
        cout << g_model.ws[i] << "  ";
    }
  }
  cout << "\n";
}

float calc_auc(const vector<Sample>& samples) {
    int32_t len = samples.size();
    vector<Pair> pairs;
    pairs.reserve(len);
    for (int i=0; i<len; ++i) {
        const Sample& s = samples[i];
        float sum = 0;
        for (int j=0; j<s.feats.size(); ++j) {
            int32_t index = s.feats[j].index;
            float val = s.feats[j].value;
            sum += g_model.ws[index] * val;
        }

        sum += g_model.bias;

        Pair p;
        p.y_pred = sigmoid(sum);
        p.label = s.label;
        pairs.push_back(p);
    }

    sort(pairs.begin(), pairs.end(), comp);

    int P = 0;
    int N = 0;
    int p_rank_sum = 0;
    for (int i=0; i<len; ++i) {
        Pair& p = pairs[i];
        if (p.label == 1) {
            ++P;
            p_rank_sum += i+1;
        } else {
            ++N;
        }
    }

    float auc = (float)(p_rank_sum - P * (P + 1) / 2) / (float)(P * N);
    return auc;
}

void calc_train_data_auc() {
    float auc = calc_auc(g_samples);
    cout << "train auc:" << auc << "\t";
}

void calc_test_data_auc() {
    float auc = calc_auc(g_test_samples);
    cout << "test auc:" << auc << endl;
}

void update_once(vector<float> deltas, int32_t from, int32_t to) {
    for (int i=from; i<to; ++i) {
        bool flag = false;
        if (is_multiple(i, NUMBER)) {
            flag = true;
        }
        Sample& s = g_samples[i];
        float delta = deltas[i - from] * g_learning_rate;

        if (flag) cout << "sample:" << i << ", delta:" << delta << "\t";
        for (int j=0; j<s.feats.size(); ++j) {
            int32_t& index = s.feats[j].index;
            float& val = s.feats[j].value;

            g_model.ws[index] += delta * val;

            if (flag && j < 4) cout << "val:" << val << ", weight:" << g_model.ws[index] << "\t";
        }
        g_model.bias += delta;
        if (flag) cout << "bias:" << g_model.bias << "\n";
    }
}

void train_once() {
    int32_t len = g_samples.size();
    int32_t batch = 1000;
    vector<float> deltas;
    int32_t from = 0;
    for (int i=0; i<len; ++i) {
        if (i % batch == 0 && i > 0) {
            update_once(deltas, from, i);
            deltas.clear();
            from = i;
        }
        Sample& s = g_samples[i];
        float sum = 0;
        for (int j=0; j<s.feats.size(); ++j) {
            int32_t& index = s.feats[j].index;
            float& val = s.feats[j].value;

            sum += g_model.ws[index] * val;
        }

        sum += g_model.bias;

        float y_pred = sigmoid(sum);
        float delta = (float)s.label - y_pred;
        if (is_multiple(i, NUMBER)) {
            cout << "sample:" << i << "Label:" << s.label << ", pred:" << y_pred << endl;
        }
        deltas.push_back(delta);
    }

    update_once(deltas, from, len);
}

void l2_reg() {
  for (int i=0; i<g_model.ws.size(); ++i) {
    if (use_feat(i)) {
      g_model.ws[i] -= l2 * g_model.ws[i];
    }
  }
  g_model.bias -= l2 * g_model.bias;
}

int main(int argc, char* argv[]) {
    cout << "init data...\n";
    init_data();

    cout << "init data...\n";
    init_model();

    for (int i=0; i<500; ++i) {
        train_once();
        l2_reg();
        if (true || i % 5 == 0 && i > 0) {
            calc_train_data_auc();
            calc_test_data_auc();
            output_model();
        }
    }
}
