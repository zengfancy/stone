/*
试图将样本特征进行加权平均，以扩大样本数量，结果auc只能到达0.60左右，远远不及原始样本
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

struct PosDelta {
    PosDelta(int32_t _i, float _delta) : 
        i(_i),
        delta(_delta) {
        }

    int32_t i;
    float delta;
};

struct NegDelta {
    NegDelta(int32_t _i1, int32_t _i2, float _delta) : 
        i1(_i1),
        i2(_i2),
        delta(_delta) {
        }

    int32_t i1;
    int32_t i2;
    float delta;
};

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

float calc_y_pred(const Sample& s) {
    float sum = 0;
    for (int j=0; j<s.feats.size(); ++j) {
        int32_t index = s.feats[j].index;
        float val = s.feats[j].value;

        sum += g_model.ws[index] * val;
    }

    sum += g_model.bias;

    return sum;
}

float calc_auc(const vector<Sample>& samples) {
    int32_t len = samples.size();
    vector<Pair> pairs;
    pairs.reserve(len);
    for (int i=0; i<len; ++i) {
        const Sample& s = samples[i];
        float sum = calc_y_pred(s);

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

void update_once(const vector<NegDelta>& deltas) {
    for (int i=0; i<deltas.size(); ++i) {
        Sample& s1 = g_samples[deltas[i].i1];
        Sample& s2 = g_samples[deltas[i].i2];
        float delta = deltas[i].delta * g_learning_rate;

        for (int j=0; j<s1.feats.size(); ++j) {
            int32_t& index = s1.feats[j].index;
            float& val = s1.feats[j].value;

            g_model.ws[index] += 0.5 * delta * val;

        }
        for (int j=0; j<s2.feats.size(); ++j) {
            int32_t& index = s2.feats[j].index;
            float& val = s2.feats[j].value;

            g_model.ws[index] += 0.5 * delta * val;

        }
        g_model.bias += delta;
    }
}

void update_once(const vector<PosDelta>& deltas) {
    for (int i=0; i<deltas.size(); ++i) {
        Sample& s = g_samples[deltas[i].i];
        float delta = deltas[i].delta * g_learning_rate;

        for (int j=0; j<s.feats.size(); ++j) {
            int32_t& index = s.feats[j].index;
            float& val = s.feats[j].value;

            g_model.ws[index] += delta * val;

        }
        g_model.bias += delta;
    }
}

void train_once() {
    int32_t len = g_samples.size();
    int32_t batch = 1000;
    vector<NegDelta> neg_deltas;
    vector<PosDelta> pos_deltas;
    for (int j=0; j<len; ++j) {
        if (j % batch == 0 && j > 0) {
            update_once(pos_deltas);
            update_once(neg_deltas);
            pos_deltas.clear();
            neg_deltas.clear();
        }
        int32_t i1 = rand() % len;
        int32_t i2 = rand() % len;
        Sample& s1 = g_samples[i1];
        Sample& s2 = g_samples[i2];

        if (s1.label == 0 || s2.label == 0) {
            float y_pred = sigmoid(0.5 * calc_y_pred(s1) + 0.5 * calc_y_pred(s2));
            float delta = -y_pred;
            neg_deltas.push_back(NegDelta(i1, i2, delta));
        } else {
            Sample& s = s1;
            int i = i1;
            if (s2.label == 1) {
                s = s2;
                i = i2;
            }

            float y_pred = sigmoid(calc_y_pred(s));
            float delta = 1 - y_pred;
            pos_deltas.push_back(PosDelta(i, delta));
        }
    }

    update_once(pos_deltas);
    update_once(neg_deltas);
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
        calc_train_data_auc();
        calc_test_data_auc();
    }
}
