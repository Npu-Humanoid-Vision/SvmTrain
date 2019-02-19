#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define POS_LABLE 1
#define NEG_LABLE 0

#ifdef __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#endif

#ifdef __WIN32
#include <io.h>
#include <windows.h>
#endif

#define MODEL_NAME "ball_nu_svc_linear_v1.xml"

// 获得某文件夹下所有图片的名字
void GetImgNames(string root_path, std::vector<std::string>& names) {
#ifdef __linux__
    struct dirent* filename;
    DIR* dir;
    dir = opendir(root_path.c_str());  
    if(NULL == dir) {  
        return;  
    }  

    int iName=0;
    while((filename = readdir(dir)) != NULL) {  
        if( strcmp( filename->d_name , "." ) == 0 ||
            strcmp( filename->d_name , "..") == 0)
            continue;

        string t_s(filename->d_name);
        names.push_back(t_s);
    }
#endif

#ifdef __WIN32
    intptr_t hFile = 0;
    struct _finddata_t fileinfo;
    string p;

    hFile = _findfirst(p.assign(root_path).append("/*").c_str(), &fileinfo);

    if (hFile != -1) {
        do {
            if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0) {
                continue;
            }
            names.push_back(fileinfo.name); 
        } while (_findnext(hFile, &fileinfo) == 0);
    }
#endif
}


void GetXsSampleData(const string folder_path, int lable, 
            cv::Mat& train_data, cv::Mat& train_data_lables) {

    // get the image names
    std::vector<std::string> image_names;
    GetImgNames(folder_path, image_names);

    // define hog descriptor 
    cv::HOGDescriptor hog_des(Size(32, 32), Size(16, 16), Size(2, 2), Size(8, 8), 12);

    // read images and compute
    for (auto i = image_names.begin(); i != image_names.end(); i++) {
        string t_path = folder_path + (*i);
        cv::Mat t_image = cv::imread(t_path);
        std::vector<float> t_descrip_vec;
        hog_des.compute(t_image, t_descrip_vec);


        // copy t_descrip_vec to train_data
        cv::Mat t_mat = cv::Mat(1, t_descrip_vec.size(), CV_32FC1);
        for (auto j = 0; j < t_descrip_vec.size(); j++) {
            t_mat.at<float>(0, j) = t_descrip_vec[j];
        }
        train_data.push_back(t_mat);
        train_data_lables.push_back(lable);
    }
}

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

    int test_sample_num = test_data.rows;
    for (auto i = 0; i < test_sample_num; i++) {
        cv::Mat test_vec = test_data.row(i);
        int t_predict_lable = (int)tester.predict(test_vec);
        if (t_predict_lable == lable) {
            true_num++;
        }
        else {
            false_num++;
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
    cout<<"test data size: "<<test_data_pos.size()+test_data_neg.size()<<endl;
    cout<<"test data lable size: "<<test_data_lables.size()<<endl;

    // get classifier 
    CvSVM tester;
    tester.load(MODEL_NAME);

    // get XX
    GetXX(test_data_pos, tester, POS_LABLE, scores.TP, scores.FN);
    GetXX(test_data_neg, tester, NEG_LABLE, scores.TN, scores.FP);

    scores.precision_rate = 1.0*scores.TP/(scores.TP+scores.FP);
    scores.recall_rate    = 1.0*scores.TP/(scores.TP+scores.FN); 
    scores.f1_score       = 2.0*scores.precision_rate*scores.recall_rate/(scores.precision_rate+scores.recall_rate);

    return ;
}

int main () {
    string test_data_path = "../../BackUpSource/Ball/Train/";
    EvaluationValues scores;
    GetScores(test_data_path, scores);
    cout<<"precision rate: \t"<<scores.precision_rate<<endl
        <<"recall rate: \t\t"<<scores.recall_rate<<endl
        <<"f1 scores: \t\t"<<scores.f1_score<<endl;
    return 0;
}