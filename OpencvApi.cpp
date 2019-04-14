#include "Params.h"


int main() {
    string pos_root_path = string(TRAINSET_PATH) + string("Pos/");
    string neg_root_path = string(TRAINSET_PATH) + string("Neg/");

    std::vector<std::string> pos_paths; 
    std::vector<std::string> neg_paths;
    GetImgNames(pos_root_path, pos_paths);
    GetImgNames(neg_root_path, neg_paths);

    std::vector<cv::Mat> image;
    std::vector<int> lable;

    for (auto i = pos_paths.begin(); i != pos_paths.end(); i++) {
        // cout<<pos_root_path+*i<<endl;
        cv::Mat t_image = cv::imread(pos_root_path+*i);
        t_image = GetUsedChannel(t_image, L);
        image.push_back(t_image);
        lable.push_back(POS_LABLE);
    }
    for (auto i = neg_paths.begin(); i != neg_paths.end(); i++) {
        cv::Mat t_image = cv::imread(neg_root_path+*i);
        t_image = GetUsedChannel(t_image, L);
        image.push_back(t_image);        
        lable.push_back(NEG_LABLE);
    }

    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(image, lable);


    pos_paths.resize(0);
    neg_paths.resize(0);
    pos_root_path = string(TESTSET_PATH) + string("Pos/");
    neg_root_path = string(TESTSET_PATH) + string("Neg/");

    GetImgNames(pos_root_path, pos_paths);
    GetImgNames(neg_root_path, neg_paths);

    int correct_sum = 0;
    for (auto i = pos_paths.begin(); i != pos_paths.end(); i++) {
        // cout<<pos_root_path+*i<<endl;
        cv::Mat t_image = cv::imread(pos_root_path+*i);
        t_image = GetUsedChannel(t_image, L);
        correct_sum += model->predict(t_image);
    }
    cout<<correct_sum*1.0/pos_paths.size()<<endl;

    correct_sum = 0;
    for (auto i = neg_paths.begin(); i != neg_paths.end(); i++) {
        cv::Mat t_image = cv::imread(neg_root_path+*i);
        t_image = GetUsedChannel(t_image, L);
        correct_sum += model->predict(t_image);
    } 
    cout<<1 - correct_sum*1.0/neg_paths.size()<<endl;

    return 0;
}