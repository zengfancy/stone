/*
 *
 * 加上l1正则化之后，能跑到0.6618左右
 *
train auc:0.540195      test auc:0.539058
train auc:0.57455       test auc:0.57232
train auc:0.612751      test auc:0.609114
train auc:0.638076      test auc:0.633379
train auc:0.65186       test auc:0.647611
train auc:0.658864      test auc:0.654497
train auc:0.662876      test auc:0.657627
train auc:0.665523      test auc:0.659301
train auc:0.667386      test auc:0.660236
train auc:0.668865      test auc:0.660821
train auc:0.670075      test auc:0.661162
train auc:0.67111       test auc:0.661375
train auc:0.672029      test auc:0.661517
train auc:0.672861      test auc:0.661607
train auc:0.67363       test auc:0.661673
train auc:0.674345      test auc:0.661726
train auc:0.675021      test auc:0.661739
train auc:0.675659      test auc:0.661755
train auc:0.676267      test auc:0.661773
train auc:0.676847      test auc:0.661767
train auc:0.677399      test auc:0.661775
train auc:0.67793       test auc:0.661763
train auc:0.678438      test auc:0.661749
train auc:0.678927      test auc:0.661737
train auc:0.679397      test auc:0.661718
train auc:0.679854      test auc:0.661684
train auc:0.680291      test auc:0.66165
train auc:0.680713      test auc:0.661628
train auc:0.681119      test auc:0.661598
train auc:0.681511      test auc:0.661563
train auc:0.68189       test auc:0.661539
 *
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

float g_learning_rate = 0.0008f;
float l1 = 0.00003f;
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
      g_model.ws[i] -= (g_model.ws[i] > 0 ? l1 : -l1);
    }
  }
  g_model.bias -= l2 * g_model.bias;
  g_model.bias -= (g_model.bias > 0 ? l1 : -l1);
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
