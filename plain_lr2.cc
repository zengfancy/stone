/*

经过了特征处理, l2 regularization 做的不对，还需改进
# map 1: x  ->  [0, 100]  ->  [30, 100]
# map 2: x  ->  [0, 50]  ->  [15, 50]
# map 11:x  ->  [0, 30]  ->  [10, 30]


train auc:0.527755      test auc:0.526605
train auc:0.530563      test auc:0.529467
train auc:0.534592      test auc:0.533504
train auc:0.539962      test auc:0.538773
train auc:0.546694      test auc:0.54534
train auc:0.554718      test auc:0.553115
train auc:0.563807      test auc:0.561882
train auc:0.57356       test auc:0.571262
train auc:0.583533      test auc:0.580972
train auc:0.593278      test auc:0.590534
train auc:0.602485      test auc:0.599383
train auc:0.610947      test auc:0.60733
train auc:0.618588      test auc:0.61443
train auc:0.625325      test auc:0.620755
train auc:0.631194      test auc:0.626316
train auc:0.636273      test auc:0.631211
train auc:0.640664      test auc:0.635622
train auc:0.644474      test auc:0.639482
train auc:0.647704      test auc:0.642763
train auc:0.650477      test auc:0.645626
train auc:0.652832      test auc:0.648053
train auc:0.654836      test auc:0.650071
train auc:0.656531      test auc:0.65179
train auc:0.657984      test auc:0.653203
train auc:0.659256      test auc:0.654328
train auc:0.660365      test auc:0.655274
train auc:0.661342      test auc:0.656046
train auc:0.662228      test auc:0.656693
train auc:0.663038      test auc:0.657255
train auc:0.663772      test auc:0.65775
train auc:0.664444      test auc:0.658174
train auc:0.665048      test auc:0.658549
train auc:0.665605      test auc:0.658863
train auc:0.666115      test auc:0.659152
train auc:0.666585      test auc:0.659392
train auc:0.667028      test auc:0.659613
train auc:0.667448      test auc:0.659803
train auc:0.667842      test auc:0.659985
train auc:0.668216      test auc:0.660143
train auc:0.668574      test auc:0.66029
train auc:0.668918      test auc:0.660429
train auc:0.66925       test auc:0.660538
train auc:0.66957       test auc:0.660648
train auc:0.669884      test auc:0.66074
train auc:0.670183      test auc:0.660829
train auc:0.670471      test auc:0.660907
train auc:0.670746      test auc:0.66098
train auc:0.671011      test auc:0.661044
train auc:0.671269      test auc:0.661101
train auc:0.671517      test auc:0.661148
train auc:0.671758      test auc:0.661186
train auc:0.671993      test auc:0.66123
train auc:0.672223      test auc:0.661268
train auc:0.672447      test auc:0.661303
train auc:0.672667      test auc:0.661333
train auc:0.672882      test auc:0.661367
train auc:0.67309       test auc:0.661391
train auc:0.673295      test auc:0.661417
train auc:0.673495      test auc:0.661444
train auc:0.673689      test auc:0.661464
train auc:0.673879      test auc:0.661484
train auc:0.674065      test auc:0.661501
train auc:0.67425       test auc:0.661519
train auc:0.674432      test auc:0.661534
train auc:0.674609      test auc:0.661548
train auc:0.674786      test auc:0.661567
train auc:0.67496       test auc:0.661574
train auc:0.675132      test auc:0.661586
train auc:0.675299      test auc:0.661595
train auc:0.675465      test auc:0.6616
train auc:0.675628      test auc:0.661601
train auc:0.675786      test auc:0.661605
train auc:0.675943      test auc:0.661612
train auc:0.676099      test auc:0.661624
train auc:0.676254      test auc:0.661629
train auc:0.676406      test auc:0.661638
train auc:0.676555      test auc:0.661641
train auc:0.676704      test auc:0.661646
train auc:0.676848      test auc:0.661645
train auc:0.676993      test auc:0.661648
train auc:0.677135      test auc:0.661645
train auc:0.677276      test auc:0.661646
train auc:0.677415      test auc:0.661642
train auc:0.677551      test auc:0.661639
train auc:0.677686      test auc:0.661648
train auc:0.67782       test auc:0.661645
train auc:0.677955      test auc:0.661643
train auc:0.678089      test auc:0.661645
train auc:0.67822       test auc:0.661645
train auc:0.678348      test auc:0.661645
train auc:0.678476      test auc:0.661634
train auc:0.678602      test auc:0.661631
train auc:0.678727      test auc:0.661628
train auc:0.678852      test auc:0.661622

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

float g_learning_rate = 0.0002f;
float l1 = 0.0001f;
float l2 = 0.0000000f;

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

            g_model.ws[index] += delta * val - l2 * g_model.ws[index];

            if (flag && j < 4) cout << "val:" << val << ", weight:" << g_model.ws[index] << "\t";
        }
        g_model.bias += delta - l2 * g_model.bias;
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

int main(int argc, char* argv[]) {
    cout << "init data...\n";
    init_data();

    cout << "init data...\n";
    init_model();

    for (int i=0; i<500; ++i) {
        train_once();
        if (true || i % 5 == 0 && i > 0) {
            calc_train_data_auc();
            calc_test_data_auc();
            output_model();
        }
    }
}
