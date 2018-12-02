/*

修改了特征处理的过程，auc有所下降
# map 1: x  ->  [0, 100]  ->  [15, 100]
# map 2: x  ->  [0, 50]  ->  [7, 50]
# map 11:x  ->  [0, 50]  ->  [7, 50]

train auc:0.529165      test auc:0.528039
train auc:0.544145      test auc:0.542516
train auc:0.567513      test auc:0.565032
train auc:0.594086      test auc:0.59096
train auc:0.616895      test auc:0.612571
train auc:0.633518      test auc:0.628771
train auc:0.64442       test auc:0.640267
train auc:0.651577      test auc:0.647948
train auc:0.656036      test auc:0.65253
train auc:0.6589        test auc:0.655509
train auc:0.661044      test auc:0.657195
train auc:0.662682      test auc:0.658189
train auc:0.66401       test auc:0.658949
train auc:0.665144      test auc:0.659519
train auc:0.666128      test auc:0.65995
train auc:0.667019      test auc:0.660273
train auc:0.667846      test auc:0.660535
train auc:0.668615      test auc:0.660718
train auc:0.669345      test auc:0.66087
train auc:0.670023      test auc:0.660985
train auc:0.670658      test auc:0.661086
train auc:0.671262      test auc:0.661158
train auc:0.67184       test auc:0.661223
train auc:0.672391      test auc:0.661275
train auc:0.672922      test auc:0.661315
train auc:0.673432      test auc:0.66135
train auc:0.673926      test auc:0.661382
train auc:0.674405      test auc:0.661413
train auc:0.674867      test auc:0.661443
train auc:0.675315      test auc:0.661465
train auc:0.675749      test auc:0.661487
train auc:0.676169      test auc:0.661512
train auc:0.676581      test auc:0.661517
train auc:0.676981      test auc:0.66153
train auc:0.677374      test auc:0.661534
train auc:0.677752      test auc:0.661539
train auc:0.678121      test auc:0.661539
train auc:0.678484      test auc:0.661537
train auc:0.67884       test auc:0.661533
train auc:0.679189      test auc:0.661529
train auc:0.679529      test auc:0.661518
train auc:0.679861      test auc:0.661501
train auc:0.680185      test auc:0.661491
train auc:0.680506      test auc:0.661486
train auc:0.680818      test auc:0.661468
train auc:0.681126      test auc:0.661457
train auc:0.681428      test auc:0.661448

之后修改成
# map 1: x  ->  [0, 100]  ->  [50, 100]
# map 2: x  ->  [0, 50]  ->  [25, 50]
# map 11:x  ->  [0, 50]  ->  [25, 50]

auc 最多只能到达0.6613左右
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

string g_file_path = "mod_libsvm2.data";

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
