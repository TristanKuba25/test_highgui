#include <opencv2/opencv.hpp>
#include <ZXing/ReadBarcode.h>
#include <ZXing/Barcode.h>
#include <ZXing/ReaderOptions.h>
#include <ZXing/BarcodeFormat.h>
#include <ZXing/Error.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include <dlfcn.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <cstring>
#include <thread>
#include <cstdio>



int  args_width  = 640;
int  args_height = 480;
bool use_npu     = true;


const std::string MODEL_PATH    = "models/ssd_mobilenet_v2.tflite";
const std::string delegate_path = "/usr/lib/libvx_delegate.so";

const float DETECTION_THRESHOLD = 0.3f;
const auto  FORMAT_MASK         = ZXing::BarcodeFormat::QRCode |
                                  ZXing::BarcodeFormat::Aztec;

const cv::Scalar COLOR_YEL(0,255,255), COLOR_GRN(0,255,0),COLOR_AIM(0,0,0);

int nb_frames_detected, frames_decoding;

 //CPU %

static uint64_t prev_idle=0, prev_tot=0;
double cpu_pct()
{
    std::ifstream f("/proc/stat");
    std::string tok; uint64_t v[10]{};
    f>>tok; for(int i=0;i<10;++i) f>>v[i];
    uint64_t idle=v[3]+v[4];
    uint64_t tot =std::accumulate(v,v+8,uint64_t{0});

    double pct = (tot!=prev_tot)
        ? 100.0*double((tot-prev_tot)-(idle-prev_idle))/double(tot-prev_tot) : 0.0;
    prev_idle=idle; prev_tot=tot;
    return std::clamp(pct,0.0,100.0);
}


struct VxDelegate {
    void* handle=nullptr;
    TfLiteDelegate* ptr=nullptr;
    using Create = TfLiteDelegate* (*)(const char* const*, const char* const*, size_t);
    using Destroy= void           (*)(TfLiteDelegate*);
    Destroy destroy=nullptr;
    explicit VxDelegate(const std::string& lib)
    {
        handle = dlopen(lib.c_str(), RTLD_LAZY|RTLD_LOCAL);
        if(!handle) throw std::runtime_error(dlerror());
        auto create = reinterpret_cast<Create>(dlsym(handle,"tflite_plugin_create_delegate"));
        destroy = reinterpret_cast<Destroy>(dlsym(handle,"tflite_plugin_destroy_delegate"));
        if(!create||!destroy) throw std::runtime_error("symbols missing");

        static const char* keys[]   = {};
        static const char* values[] = {};
        ptr = create(keys, values, 0);
        if(!ptr) throw std::runtime_error("delegate create failed");
    }
    ~VxDelegate(){ if(ptr&&destroy) destroy(ptr); if(handle) dlclose(handle); }
};


std::unique_ptr<tflite::Interpreter> interpreter;
int input_index, out_boxes,out_classes,out_scores,out_count;
int INPUT_H,INPUT_W;

// Utils
cv::Mat preprocess_frame(const cv::Mat& img,cv::Size sz)
{ cv::Mat r; cv::resize(img,r,sz); return r; }

cv::Mat binarize_image(const cv::Mat& img)
{
    cv::Mat g,bin; cv::cvtColor(img,g,cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(g,bin,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,11,2);
    return bin;
}
cv::Mat sharpen_image(const cv::Mat& img)
{
    static const cv::Mat k=(cv::Mat_<char>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);
    cv::Mat dst; cv::filter2D(img,dst,CV_8U,k); return dst;
}


using Obj = std::map<std::string,float>;
std::vector<Obj> detect_objects(const cv::Mat& inp,float thr)
{
    std::memcpy(interpreter->typed_tensor<uint8_t>(input_index),inp.data,INPUT_W*INPUT_H*3);
    interpreter->Invoke();

    const float* b = interpreter->typed_tensor<float>(out_boxes);
    const float* c = interpreter->typed_tensor<float>(out_classes);
    const float* s = interpreter->typed_tensor<float>(out_scores);
    int n = static_cast<int>(interpreter->typed_tensor<float>(out_count)[0]);

    std::vector<Obj> res;
    for(int i=0;i<n;++i){
        if(s[i]<thr) continue;
        res.push_back({
            {"ymin",b[4*i+0]}, {"xmin",b[4*i+1]},
            {"ymax",b[4*i+2]}, {"xmax",b[4*i+3]},
            {"class_id",c[i]}, {"score",s[i]}
        });
        /*printf("Detected object %d: [%.2f,%.2f,%.2f,%.2f] class %d score %.2f\n",
               i, b[4*i+0], b[4*i+1], b[4*i+2], b[4*i+3], int(c[i]), s[i]);*/
    }
    return res;
}

static const std::string ERR_DIR = "/root/zxing_test/official_lib/mobilenet_v2/images_save/";

// Create the error directory if it does not exist
inline void init_error_dir()
{
    std::error_code ec;
    std::filesystem::create_directories(ERR_DIR, ec);
    if (ec)
        std::cerr << "cannot create " << ERR_DIR << " : " << ec.message() << '\n';
}


static int err_id = 0;


void save_error(const cv::Mat& img, const std::string& prefix, const ZXing::Error& err)
{
    char fname[256];

    std::snprintf(fname, sizeof(fname), "%s%s_%s_%04d.jpg", ERR_DIR.c_str(), prefix.c_str(),(ZXing::ToString(err.type())).c_str(), err_id);
    /*std::snprintf(fname, sizeof(fname),
              "%s%s_%s_%04d.png", ERR_DIR.c_str(), prefix.c_str(),
              ZXing::ToString(err.type()).c_str(), err_id);*/

    if (cv::imwrite(fname, img))
        std::cout << "[ERR] image saved : " << fname << '\n';
    else
        std::cerr << "[ERR] could not save : " << fname << '\n';

    ++err_id;
}



cv::Mat run_odt_and_draw_results_frame(cv::Mat frame)
{
    cv::Mat inp = preprocess_frame(frame,{INPUT_W,INPUT_H});
    auto objs = detect_objects(frame,DETECTION_THRESHOLD);

    for(const auto& o:objs){
        int xmin=int(o.at("xmin")*frame.cols);
        int xmax=int(o.at("xmax")*frame.cols);
        int ymin=int(o.at("ymin")*frame.rows);
        int ymax=int(o.at("ymax")*frame.rows);
        xmin=std::clamp(xmin,0,frame.cols-1);
        xmax=std::clamp(xmax,0,frame.cols-1);
        ymin=std::clamp(ymin,0,frame.rows-1);
        ymax=std::clamp(ymax,0,frame.rows-1);
        cv::rectangle(frame,{xmin,ymin,xmax-xmin,ymax-ymin},COLOR_YEL,2);
       /* cv::putText(frame,cv::format("%.1f%%",o.at("score")*100.0f),
                    {xmin,ymin-10},cv::FONT_HERSHEY_SIMPLEX,0.5,COLOR_YEL,2);*/

        cv::Mat roi_gray;
        cv::cvtColor(frame(cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin)),roi_gray,cv::COLOR_BGR2GRAY);

        ZXing::ReaderOptions opts;
        opts.setFormats(FORMAT_MASK).setTryDownscale(true).setReturnErrors(true);

        double tbegin=cv::getTickCount();

        auto res = ZXing::ReadBarcodes(ZXing::ImageView(roi_gray.data,roi_gray.cols,roi_gray.rows,ZXing::ImageFormat::Lum,roi_gray.step),opts);
        double tend=cv::getTickCount();
        double zx_ms = (tend-tbegin)/cv::getTickFrequency()*1000.0;

        for(const auto& r:res){
            nb_frames_detected = nb_frames_detected + 1;
            printf("nb_frames_detected : %d\n",nb_frames_detected);
            printf("frames_decoding : %d\n",frames_decoding);
            if (nb_frames_detected ==100){
                printf("100 frames detected, ratio : %d\n", (frames_decoding*100)/nb_frames_detected);
                nb_frames_detected = 0;
                frames_decoding = 0;
            }


            if (r.isValid()){
                frames_decoding = frames_decoding + 1;
                auto p=r.position();
                if (!r.text().empty()) {
                    std::vector<cv::Point> pts={
                        {p.topLeft().x,p.topLeft().y},
                        {p.topRight().x,p.topRight().y},
                        {p.bottomRight().x,p.bottomRight().y},
                        {p.bottomLeft().x,p.bottomLeft().y}};
                    cv::polylines(frame,pts,true,COLOR_GRN,2);
                    std::cout<<'['<<args_width<<'x'<<args_height<<"] Decoded Text in "<<std::fixed<<std::setprecision(1)<<zx_ms<<" ms using ML : "
                            <<r.text()<<" ("<<ZXing::ToString(r.format())<<")\n";
                    break;
                    }
                }else {
                const ZXing::Error& err = r.error();
                save_error(roi_gray, "roi", err);
                }
        }

    }
    return frame;
}




int main(int argc,char* argv[])
{
    // Parsing command line arguments
    for(int i=1;i<argc;++i){
        if((!strcmp(argv[i],"-w")||!strcmp(argv[i],"--width"))&&i+1<argc)
            args_width=std::stoi(argv[++i]);
        else if((!strcmp(argv[i],"-H")||!strcmp(argv[i],"--height"))&&i+1<argc)
            args_height=std::stoi(argv[++i]);
        //else if(!strcmp(argv[i],"--npu")) use_npu=true;
    }
    std::cout<<"[INFO] "<<args_width<<"x"<<args_height<<"\n";

    setenv("TFLITE_DISABLE_DEFAULT_DELEGATES","1",1);
    auto model=tflite::FlatBufferModel::BuildFromFile(MODEL_PATH.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model,resolver)(&interpreter);
    if(!interpreter){ std::cerr<<"Interpreter build fail\n"; return 1; }

    const int num_threads = std::max(1u,std::thread::hardware_concurrency());

    std::unique_ptr<VxDelegate> vx;
    bool vx_ok=false;
    if(use_npu){
        try{
            vx = std::make_unique<VxDelegate>(delegate_path);
            if(interpreter->ModifyGraphWithDelegate(vx->ptr)==kTfLiteOk){
                std::cout<<"NPU delegate active\n";
                vx_ok=true;
            }else{
                std::cerr<<"VX refused, fallback CPU\n";
                vx.reset();
            }
        }catch(const std::exception& e){
            std::cerr<<"VX init failed ("<<e.what()<<"), CPU mode\n";
            vx.reset();
        }
    }

    TfLiteDelegate* xnn=nullptr;
    if(!vx_ok){
        TfLiteXNNPackDelegateOptions xopts = TfLiteXNNPackDelegateOptionsDefault();
        xopts.num_threads = num_threads;
        xnn = TfLiteXNNPackDelegateCreate(&xopts);
        if(interpreter->ModifyGraphWithDelegate(xnn)==kTfLiteOk)
            std::cout<<"XNNPACK active ("<<num_threads<<" thr)\n";
        else {
            std::cerr<<"No delegate\n"; return 1;
        }
        interpreter->SetNumThreads(num_threads);
    }

    interpreter->AllocateTensors();
    input_index = interpreter->inputs()[0];
    out_boxes   = interpreter->outputs()[0];
    out_classes = interpreter->outputs()[1];
    out_scores  = interpreter->outputs()[2];
    out_count   = interpreter->outputs()[3];
    INPUT_H = interpreter->tensor(input_index)->dims->data[1];
    INPUT_W = interpreter->tensor(input_index)->dims->data[2];


    cv::VideoCapture cap;
    if(!cap.open(3,cv::CAP_V4L2)){ std::cerr<<"Cam open fail\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH ,args_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,args_height);

    nb_frames_detected=0;
    frames_decoding=0;

    std::vector<double> fts,cpu; const int AVG=10;
    cv::Mat frame;
    while(true){
        double t0=cv::getTickCount();
        if(!cap.read(frame)||frame.empty()) break;

        frame = run_odt_and_draw_results_frame(frame);
        cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::flip(frame,frame,1);


        // Target aim overlay
        cv::putText(frame,"Fit your Qr code in",{frame.cols/2-85,frame.rows/2-10},cv::FONT_HERSHEY_SIMPLEX,0.7,COLOR_AIM,2);
        int cx=frame.cols/2, cy=frame.rows/2, L=40,SZ=170;
        cv::line(frame,{cx-L,cy},{cx+L,cy},COLOR_AIM,2);
        cv::line(frame,{cx,cy-L},{cx,cy+L},COLOR_AIM,2);
        cv::rectangle(frame,{cx-SZ,cy-SZ},{cx+SZ,cy+SZ},COLOR_AIM,2);

        fts.push_back((cv::getTickCount()-t0)/cv::getTickFrequency());
        cpu.push_back(cpu_pct());
        if(fts.size()==AVG){
            double fps=AVG/std::accumulate(fts.begin(),fts.end(),0.0);
            double c  =std::accumulate(cpu.begin(),cpu.end(),0.0)/AVG;
            std::cout<<std::fixed<<std::setprecision(1) <<"[INFO] Mean FPS (each "<<AVG<<" frames) : "<<fps<<" | CPU "<<c<<"%\n";
            fts.clear(); cpu.clear();
        }

        cv::imshow("result",frame);
        int k=cv::waitKey(1);
        if(k=='q'||k==27) break;
    }
    return 0;
}
