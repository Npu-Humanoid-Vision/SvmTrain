#include "HogGetter.h"
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define POS_LABLE 1
#define NEG_LABLE 0

void GetXsSampleData(const string& folder_path, const string& postfix, int lable, 
            cv::Mat& train_data, cv::Mat& train_data_lables) {
    
    HogGetter hog_getter;
    hog_getter.ImageReader_(folder_path, postfix);
    hog_getter.HogComputter_();

    // train_data = hog_getter.sample_features_.clone();
    train_data.push_back(hog_getter.sample_features_.clone());
    for (int i = 0; i < hog_getter.sample_nums_; i++) { // 
        train_data_lables.push_back(lable);
    }
}

int main(int argc, char const *argv[]) {

    string root_path = "../../BackUpSource/People/Train/";
    cv::Mat train_data;
    cv::Mat train_data_lables;

    GetXsSampleData( root_path + "NegSample/", "*.png", NEG_LABLE, train_data, train_data_lables );
    // cout<<train_data.size()<<' '<<train_data.type()<<endl;
    // cout<<train_data_lables.size()<<' '<<train_data_lables.type()<<endl;
    GetXsSampleData( root_path + "PosSample/", "*.png", POS_LABLE, train_data, train_data_lables );
    // cout<<train_data.size()<<endl;
    // cout<<train_data_lables.size()<<endl;

    // 参数设置
    CvSVMParams train_params;
    train_params.svm_type = CvSVM::C_SVC;
    train_params.kernel_type = CvSVM::LINEAR;
    train_params.degree = 0;
    train_params.gamma = 1;
    train_params.coef0 = 0;
    train_params.C = 1;
    train_params.nu = 0;
    train_params.p = 0;
    train_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 0.01); // 训练终止条件

    CvSVM trainer;
    trainer.train(train_data, train_data_lables, cv::Mat(), cv::Mat(), train_params);

    cout<<"train done"<<endl;
    trainer.save("model.xml");

    return 0;
}
