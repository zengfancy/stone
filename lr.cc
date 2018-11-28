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
    vector<float> vs;
    float bias;
} Model;

Model g_model;
vector<Sample> g_samples;
vector<Sample> g_test_samples;
int32_t g_vlen = 12;
int32_t g_app_num = 1000;

// 月访问次数
// 某天访问最大次数
// 访问天数
// 一天访问平均次数
// 第1,2,3,4,5,6周访问次数
// 月访问活跃值
// 月访问次数占比
int32_t g_used_vs[] = {1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0};
int32_t g_used_vlen = 3;

float g_learning_rate = 0.01f;
float l1 = 0.0001f;
float l2 = 0.0001f;

string g_file_path = "test.data.txt";

bool use_feat(int32_t index) {
  int32_t v = (index - 1) % g_vlen;
  return v == 1 || v == 2 || v == 11;
}

// if index == 12, w = 0, _v = 11, v = 2
void get_wv(int32_t index, int32_t& w, int32_t& v) {
  w = (index - 1) / g_vlen;
  int32_t _v = (index - 1) % g_vlen;
  v = g_used_vs[_v] - 1;
}

float sigmoid(float in) {
  return 1 / (1 + exp(-in));
}

void init_data() {
    string line;
    ifstream fs(g_file_path.c_str());

    while (getline(fs, line)) {
        Sample s;
        istringstream ls(line);
        string label_str;
        getline(ls, label_str, ' ');
        s.label = atoi(label_str.c_str());

        string feat_str;
        while (getline(ls, feat_str, ' ')) {
            size_t pos = feat_str.find(':');
            string index_str = feat_str.substr(0, pos);
            string val_str = feat_str.substr(pos + 1);

            Feat feat;
            feat.index = atoi(index_str.c_str());
            feat.value = atof(val_str.c_str());
            if (use_feat(feat.index)) {
              s.feats.push_back(feat);
            }
        }

        if (s.feats.empty()) {
          continue;
        }
        int rnum = rand() % 100;
        if (rnum < 40) {
          g_test_samples.push_back(s);
        } else {
          g_samples.push_back(s);
        }
    }

    fs.close();
    cout << "train samples:" << g_samples.size() << endl;
    cout << "test samples:" << g_test_samples.size() << endl;
}

void init_model() {
    g_model.bias = 0;

    g_model.ws.resize(g_app_num, 0.01f);
    g_model.vs.resize(g_used_vlen, 1.0f);
}

bool comp(Pair a, Pair b) {
    return a.y_pred < b.y_pred;
}

void output_model() {
  cout << "bias:" << g_model.bias << "\n";
  cout << "ws:";
  for (int i=0; i<g_model.ws.size(); ++i) {
    cout << g_model.ws[i] << "  ";
  }
  cout << "\n";
  cout << "vs:";
  for (int i=0; i<g_model.vs.size(); ++i) {
    cout << g_model.vs[i] << "  ";
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
            int32_t w_i, v_i;
            get_wv(index, w_i, v_i);
            sum += g_model.ws[w_i] * g_model.vs[v_i] * val;
        }

        sum += g_model.bias;

        Pair p;
        p.y_pred = sigmoid(sum);
        p.label = s.label;
        pairs.push_back(p);
    }

    // ascending sort
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
    cout << "train auc:" << auc << endl;
}

void calc_test_data_auc() {
    float auc = calc_auc(g_test_samples);
    cout << "test  auc:" << auc << endl;
}

void update_once(float delta_sum, int32_t from, int32_t to) {
    float delta = delta_sum * g_learning_rate;
    for (int i=from; i<to; ++i) {
        Sample& s = g_samples[i];
        for (int j=0; j<s.feats.size(); ++j) {
            int32_t& index = s.feats[j].index;
            float& val = s.feats[j].value;
            int32_t w_i, v_i;
            get_wv(index, w_i, v_i);
            g_model.ws[w_i] += delta * g_model.vs[v_i] * val - l2 * g_model.ws[w_i];
            g_model.vs[v_i] += delta * g_model.ws[w_i] * val - l2 * g_model.vs[v_i];

            g_model.bias += delta - l2 * g_model.bias;
        }
    }
}

void train_once() {
    int32_t len = g_samples.size();
    int32_t batch = 1000;
    float delta_sum = 0;
    int32_t from = 0;
    for (int i=0; i<len; ++i) {
        if (i % batch == 0 && i > 0) {
            update_once(delta_sum, from, i);
            delta_sum = 0;
            from = i;
        }
        Sample& s = g_samples[i];
        float sum = 0;
        for (int j=0; j<s.feats.size(); ++j) {
            int32_t& index = s.feats[j].index;
            float& val = s.feats[j].value;

            int32_t w_i, v_i;
            get_wv(index, w_i, v_i);
            sum += g_model.ws[w_i] * g_model.vs[v_i] * val;
        }

        sum += g_model.bias;

        float y_pred = sigmoid(sum);
        delta_sum += (float)s.label - y_pred;
    }

    update_once(delta_sum, from, len);
    delta_sum = 0;
}

int main(int argc, char* argv[]) {
    cout << "init data...\n";
    init_data();

    cout << "init data...\n";
    init_model();

    for (int i=0; i<100; ++i) {
        cout << "train step:" << i << endl;
        train_once();
        if (i % 10 == 0 && i > 0) {
            calc_train_data_auc();
            calc_test_data_auc();
            output_model();
        }
    }
}

