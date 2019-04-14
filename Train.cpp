#include "Params.h"

int main(int argc, char const *argv[]) {
    // for train time
    double begin;

    string pos_root_path = string(TRAINSET_PATH) + string("Pos/");
    string neg_root_path = string(TRAINSET_PATH) + string("Neg/");
    cout<<pos_root_path<<endl<<neg_root_path<<endl;
    cv::Mat train_data;
    cv::Mat train_data_lables;
    GetXsSampleData(pos_root_path, POS_LABLE, train_data, train_data_lables);
    GetXsSampleData(neg_root_path, NEG_LABLE, train_data, train_data_lables);
    cout<<train_data.size()<<' '<<train_data_lables.size()<<endl;

#ifdef __WIN32 // mingw 只配了 opencv2
    // 参数设置
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 50000, FLT_EPSILON);
    //SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
    CvSVMParams train_params(CvSVM::C_SVC, CvSVM::LINEAR, 0.1, 0.1, 0.1, 1, 0.5, 0.5, 0, criteria);

    CvSVM trainer;

    begin = (double)getTickCount();
    // trainer.train(train_data, train_data_lables, cv::Mat(), cv::Mat(), train_params);
    trainer.train_auto(train_data, train_data_lables, cv::Mat(), cv::Mat(), train_params);


    cout<<"train take time: "<<((double)getTickCount() - begin)/getTickFrequency()<<endl;
    cout<<"C: "<<trainer.get_params().C<<endl;
    trainer.save(MODEL_NAME);
    cout<<"Save model to: "<<MODEL_NAME<<endl;

#endif

#ifdef __linux__

    Ptr<cv::ml::SVM> trainer = cv::ml::SVM::create();
    trainer->setType(cv::ml::SVM::Types::C_SVC);
    trainer->setKernel(cv::ml::SVM::KernelTypes::RBF);
    // trainer->setC(0.1);
 
	Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, train_data_lables);
    cv::ml::ParamGrid c_g(0.0001, 1000, 10);

    begin = (double)getTickCount();
    trainer->trainAuto(tdata, 10, c_g);
    cout<<"train take time: "<<((double)getTickCount() - begin)/getTickFrequency()<<endl;
	trainer->save("ball_rbf_auto_v5.xml");//保存
    // rbf(弄错了..应该 默认的linear)
    // auto, v2 是之前的参数, 进行了 data argumentation
    // v3  hog stribe从(8,8)改为(2,2)
    // v4  v3 + hog nbins从9改为12
    // v5 同 v4 测试时间

    // linear(真正的) 
    // v1 参数是之前参数
    // v2 hog 是rbf v4 参数, 重新进行了 data argumentation
#endif
    return 0;
}
