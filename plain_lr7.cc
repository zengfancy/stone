/*
 *
 * 对特征做了缩放处理，能涨到0.6645左右
 *
 * feat_value /= sum_feat_value
 * feat_value *= sqrt(sum_feat_value)
 *
train auc:0.69211       test auc:0.664456
train auc:0.692249      test auc:0.66446
train auc:0.692388      test auc:0.664459
train auc:0.692525      test auc:0.664463
train auc:0.692662      test auc:0.664463
train auc:0.692796      test auc:0.664465
train auc:0.692929      test auc:0.664473
train auc:0.693063      test auc:0.664483
train auc:0.693193      test auc:0.664485
train auc:0.693322      test auc:0.664485
train auc:0.693451      test auc:0.664485
train auc:0.693577      test auc:0.66449
train auc:0.693702      test auc:0.664488
train auc:0.693826      test auc:0.664487
train auc:0.693949      test auc:0.664492
train auc:0.69407       test auc:0.664492
train auc:0.694191      test auc:0.664494
train auc:0.694311      test auc:0.664495
train auc:0.694428      test auc:0.66449
train auc:0.694547      test auc:0.664489
train auc:0.694665      test auc:0.664486
train auc:0.694781      test auc:0.664483
train auc:0.694897      test auc:0.664482
train auc:0.695011      test auc:0.664484
train auc:0.695123      test auc:0.664477
train auc:0.695235      test auc:0.664474
train auc:0.695346      test auc:0.664474
train auc:0.695455      test auc:0.664473
train auc:0.695565      test auc:0.664469
train auc:0.695672      test auc:0.66447
train auc:0.695779      test auc:0.664468
train auc:0.695884      test auc:0.664467
train auc:0.695989      test auc:0.664466
train auc:0.696095      test auc:0.664461
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

float g_learning_rate = 0.0040f;
float l1 = 0.00015f;
float l2 = 0.00020f;

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
        float weight_sum[12];
        for (int i=0; i<12; ++i) { weight_sum[i] = 0; }
        while (getline(ls, feat_str, ' ')) {
            size_t pos = feat_str.find(':');
            string index_str = feat_str.substr(0, pos);
            string val_str = feat_str.substr(pos + 1);

            Feat feat;
            feat.index = atoi(index_str.c_str());
            feat.value = atof(val_str.c_str()) * 0.01f;
            weight_sum[feat.index % 12] += feat.value;
            if (use_feat(feat.index)) {
              s.feats.push_back(feat);
            }
        }

        if (s.feats.empty()) {
          continue;
        }

        for (int i=0; i<s.feats.size(); ++i) {
            s.feats[i].value /= weight_sum[s.feats[i].index % 12];
            s.feats[i].value *= sqrt(weight_sum[s.feats[i].index % 12]);
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
        calc_train_data_auc();
        calc_test_data_auc();
        output_model();
    }
}
