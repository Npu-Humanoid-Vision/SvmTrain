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
    cv::HOGDescriptor hog_des(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);

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

int main(int argc, char const *argv[]) {

    string pos_root_path = "../../BackUpSource/Ball/Train/Preproc/Pos/";
    string neg_root_path = "../../BackUpSource/Ball/Train/Preproc/Neg/";
    cv::Mat train_data;
    cv::Mat train_data_lables;
    GetXsSampleData(pos_root_path, POS_LABLE, train_data, train_data_lables);
    GetXsSampleData(neg_root_path, NEG_LABLE, train_data, train_data_lables);
    cout<<train_data.size()<<' '<<train_data_lables.size()<<endl;

#ifdef __WIN32 // mingw 只配了 opencv2
    // 参数设置
    CvSVMParams train_params;
    train_params.svm_type = CvSVM::C_SVC;
    train_params.kernel_type = CvSVM::RBF;
    train_params.degree = 0;
    train_params.gamma = 1;
    train_params.coef0 = 0;
    train_params.C = 1;
    train_params.nu = 0;
    train_params.p = 0;
    train_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100000, 1e-7); // 训练终止条件

    CvSVM trainer;
    // trainer.train(train_data, train_data_lables, cv::Mat(), cv::Mat(), train_params);
    // trainer.train()
    trainer.train_auto(train_data, train_data_lables, cv::Mat(), cv::Mat(), train_params);

    // cout<<trainer.params.C<<trainer.params.nu<<endl;
    
    cout<<"train done"<<endl;
    trainer.save("ball_rbf_auto.xml");

#endif

#ifdef __linux__

#endif

    // GetXsSampleData( root_path + "Pos/", "*.jpg", POS_LABLE, train_data, train_data_lables );
    // GetXsSampleData( root_path + "Neg/", "*.jpg", NEG_LABLE, train_data, train_data_lables );
    // // cout<<train_data.size()<<' '<<train_data.type()<<endl;
    // // cout<<train_data_lables.size()<<' '<<train_data_lables.type()<<endl;
    // // cout<<train_data.size()<<endl;
    // // cout<<train_data_lables.size()<<endl;

    // // 参数设置
    // CvSVMParams train_params;
    // train_params.svm_type = CvSVM::C_SVC;
    // train_params.kernel_type = CvSVM::RBF;
    // train_params.degree = 0;
    // train_params.gamma = 1;
    // train_params.coef0 = 0;
    // train_params.C = 1;
    // train_params.nu = 0;
    // train_params.p = 0;
    // train_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100000, 1e-7); // 训练终止条件

    // CvSVM trainer;
    // // trainer.train(train_data, train_data_lables, cv::Mat(), cv::Mat(), train_params);
    // // trainer.train()
    // trainer.train_auto(train_data, train_data_lables, cv::Mat(), cv::Mat(), train_params);

    // cout<<trainer.params.C<<trainer.params.nu<<endl;
    // cout<<"train done"<<endl;
    // trainer.save("ball_linear_auto.xml");

    return 0;
}
