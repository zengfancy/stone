/*
 *
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

Model g_model1;
vector<Sample> g_samples1;
vector<Sample> g_test_samples1;
Model g_model2;
vector<Sample> g_samples2;
vector<Sample> g_test_samples2;

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
        Sample s1, s2;
        istringstream ls(line);
        string label_str;
        getline(ls, label_str, ' ');
        s1.label = s2.label = atoi(label_str.c_str());
        if (s1.label == 1) {
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
              s1.feats.push_back(feat);
              s2.feats.push_back(feat);
            }
        }

        if (s1.feats.empty()) {
          continue;
        }

        for (int i=0; i<s2.feats.size(); ++i) {
            s2.feats[i].value /= weight_sum[s2.feats[i].index % 12];
            s2.feats[i].value *= sqrt(weight_sum[s2.feats[i].index % 12]);
        }
        int rnum = rand() % 100;
        if (rnum < 20) {
          g_test_samples1.push_back(s1);
          g_test_samples1.push_back(s2);
        } else {
          g_samples1.push_back(s1);
          g_samples1.push_back(s2);
        }
    }

    fs.close();
    cout << "train samples:" << g_samples1.size() << ", P:" << P << ", N:" << N << endl;
    cout << "test samples:" << g_test_samples1.size() << ", P:" << P << ", N:" << N << endl;
}

void init_model() {
    g_model1.bias = 0;
    g_model2.bias = 0;

    g_model1.ws.resize(g_app_num * 12, -0.01f);
    g_model2.ws.resize(g_app_num * 12, -0.01f);
}

bool comp(Pair a, Pair b) {
    return a.y_pred < b.y_pred;
}

float calc_y_pred(const Sample& s, const Model& model) {
    float sum = 0;
    for (int j=0; j<s.feats.size(); ++j) {
        int32_t index = s.feats[j].index;
        float val = s.feats[j].value;

        sum += model.ws[index] * val;
    }

    sum += model.bias;

    return sum;
}

float calc_auc(const vector<Sample>& samples, const Model& model) {
    int32_t len = samples.size();
    vector<Pair> pairs;
    pairs.reserve(len);
    for (int i=0; i<len; ++i) {
        const Sample& s = samples[i];
        float sum = calc_y_pred(s, model);

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

float calc_final_auc() {
    int32_t len = g_samples1.size();
    vector<Pair> pairs;
    pairs.reserve(len);
    for (int i=0; i<len; ++i) {
        const Sample& s1 = g_samples1[i];
        float sum1 = calc_y_pred(s1, g_model1);
        const Sample& s2 = g_samples2[i];
        float sum2 = calc_y_pred(s2, g_model2);

        Pair p;
        p.y_pred = sqrt(sigmoid(sum1) * sigmoid(sum2));
        
        p.label = s1.label;
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

void update_once(vector<float> deltas, int32_t from, int32_t to,
        const vector<Sample>& samples, Model& model) {
    for (int i=from; i<to; ++i) {
        const Sample& s = samples[i];
        float delta = deltas[i - from] * g_learning_rate;

        for (int j=0; j<s.feats.size(); ++j) {
            int32_t index = s.feats[j].index;
            float val = s.feats[j].value;

            model.ws[index] += delta * val;

        }
        model.bias += delta;
    }
}

void train_once(const vector<Sample>& samples, Model& model) {
    int32_t len = samples.size();
    int32_t batch = 1000;
    vector<float> deltas;
    int32_t from = 0;
    for (int i=0; i<len; ++i) {
        if (i % batch == 0 && i > 0) {
            update_once(deltas, from, i, samples, model);
            deltas.clear();
            from = i;
        }
        const Sample& s = samples[i];
        float sum = calc_y_pred(s, model);

        float y_pred = sigmoid(sum);
        float delta = (float)s.label - y_pred;
        deltas.push_back(delta);
    }

    update_once(deltas, from, len, samples, model);
}

void l2_reg(Model& model) {
  for (int i=0; i<model.ws.size(); ++i) {
    if (use_feat(i)) {
      model.ws[i] -= l2 * model.ws[i];
      model.ws[i] -= (model.ws[i] > 0 ? l1 : -l1);
    }
  }
  model.bias -= l2 * model.bias;
  model.bias -= (model.bias > 0 ? l1 : -l1);
}

int main(int argc, char* argv[]) {
    cout << "init data...\n";
    init_data();

    cout << "init data...\n";
    init_model();

    cout << "training the first model...\n";
    for (int i=0; i<500; ++i) {
        cout << "step " << i << "\t";
        train_once(g_samples1, g_model1);
        l2_reg(g_model1);
        float auc = calc_auc(g_test_samples1, g_model1);
        cout << "test auc:" << auc << endl;
        if (auc > 0.6617) {
            break;
        }
    }

    cout << "training the second model...\n";
    for (int i=0; i<500; ++i) {
        cout << "step " << i << "\t";
        train_once(g_samples2, g_model2);
        l2_reg(g_model2);
        float auc = calc_auc(g_test_samples2, g_model2);
        cout << "test auc:" << auc << endl;
        if (auc > 0.6644) {
            break;
        }
    }

    float auc = calc_final_auc();
    cout << "final auc:" << auc << endl;
}


