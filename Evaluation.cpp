#include "Params.h"



// get test_data, return TP, FP, FN, TN num
/*
+————————————————————————————————————————————————————+
|actual               | positive   | actual negative |
|predicted positive   | TP	       | FP              |
|predicted negative   | FN	       | TN              |
+————————————————————————————————————————————————————+
*/
void GetXX(cv::Mat& test_data, CvSVM& tester, int lable, int& true_num, int& false_num) {
    true_num  = 0;
    false_num = 0;

    // load image for show
    string folder_path = TESTSET_PATH;
    if (lable == POS_LABLE) {
        folder_path += "Pos/";
    }
    else {
        folder_path += "Neg/";
    }
    std::vector<std::string> image_names;
    GetImgNames(folder_path, image_names);


    int test_sample_num = test_data.rows;
    for (auto i = 0; i < test_sample_num; i++) {
        cv::Mat test_vec = test_data.row(i);
#ifdef REGRESSION
        double scores = tester.predict(test_vec);
        int t_predict_lable;
        if (scores > 0.5) {
            t_predict_lable = POS_LABLE;
        }
        else {
            t_predict_lable = NEG_LABLE;
        }
#else
        int t_predict_lable = (int)tester.predict(test_vec);
#endif
        if (t_predict_lable == lable) {
            true_num++;
        }
        else {
            false_num++;
            cout<<"wrong classified: "<<folder_path<<image_names[i]<<endl;
            cv::Mat t = cv::imread(folder_path+image_names[i]);
            cv::imshow("wrong classified", t);
            cv::waitKey(0);
        }
    }
    return ;
}

struct EvaluationValues {
    int TP, FP;
    int FN, TN;
    double precision_rate;
    double recall_rate;
    double f1_score;
};
// 返回各种指标
void GetScores(string test_data_path, EvaluationValues& scores) {
    // get test data
    cv::Mat test_data_pos;
    cv::Mat test_data_neg;
    cv::Mat test_data_lables;
    GetXsSampleData(test_data_path+"/Pos/", POS_LABLE, test_data_pos, test_data_lables);
    GetXsSampleData(test_data_path+"/Neg/", NEG_LABLE, test_data_neg, test_data_lables);
    cout<<"test data size: "<<test_data_lables.size()<<endl;

    // get classifier 
    CvSVM tester;
    tester.load(MODEL_NAME);

    // get XX
    GetXX(test_data_pos, tester, POS_LABLE, scores.TP, scores.FN);
    GetXX(test_data_neg, tester, NEG_LABLE, scores.TN, scores.FP);
    cout<<"TP: "<<scores.TP<<"\t FP: "<<scores.FP<<endl
        <<"FN: "<<scores.FN<<"\t TN: "<<scores.TN<<endl;

    scores.precision_rate = 1.0*scores.TP/(scores.TP+scores.FP);
    scores.recall_rate    = 1.0*scores.TP/(scores.TP+scores.FN); 
    scores.f1_score       = 2.0*scores.precision_rate*scores.recall_rate/(scores.precision_rate+scores.recall_rate);

    return ;
}

int main () {
    string test_data_path = TESTSET_PATH;
    EvaluationValues scores;
    GetScores(test_data_path, scores);
    cout<<"precision rate: \t"<<scores.precision_rate<<endl
        <<"recall rate: \t\t"<<scores.recall_rate<<endl
        <<"f1 scores: \t\t"<<scores.f1_score<<endl;
    return 0;
}