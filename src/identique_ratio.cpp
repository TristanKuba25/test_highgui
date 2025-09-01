#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <cstring>
#include <thread>
#include <cstdio>

#include <opencv2/opencv.hpp>

// ZXing-C++ 
#include <ZXing/ReadBarcode.h>
#include <ZXing/Barcode.h>
#include <ZXing/ReaderOptions.h>
#include <ZXing/BarcodeFormat.h>
#include <ZXing/Error.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include "../headers/anchors_v3.h" // generated header with NUM_ANCHORS=2034 and anchors_v3[][]

int  args_width  = 640;
int  args_height = 480;
bool use_npu     = true;


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


inline float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }

struct Detection {
    cv::Rect box;   // pixel coords
    float score;    // confidence
    int class_id;   // 0..num_classes-1 (after background removal)
};


std::vector<Detection> nms(const std::vector<Detection>& dets, float iou_thresh = 0.5f)
{
    std::vector<Detection> out;
    std::vector<Detection> sorted = dets;
    std::sort(sorted.begin(), sorted.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<char> removed(sorted.size(), 0);
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (removed[i]) continue;
        out.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (removed[j]) continue;
            float inter = (sorted[i].box & sorted[j].box).area();
            float uni = sorted[i].box.area() + sorted[j].box.area() - inter;
            float iou = uni > 0.f ? inter / uni : 0.f;
            if (iou > iou_thresh) removed[j] = 1;
        }
    }
    return out;
}

struct Decoded {
    std::vector<cv::Rect2f> boxes_norm;
    std::vector<std::vector<float>> probs;
};

Decoded decoder_v3(const float* boxes_raw, const float* scores_raw,
                   int num_boxes, int num_classes_raw)
{
    Decoded dec;
    dec.boxes_norm.reserve(num_boxes);
    dec.probs.reserve(num_boxes);

    for (int i = 0; i < num_boxes; ++i) {
        float ty = boxes_raw[i*4 + 0] / 10.f;
        float tx = boxes_raw[i*4 + 1] / 10.f;
        float th = boxes_raw[i*4 + 2] / 5.f;
        float tw = boxes_raw[i*4 + 3] / 5.f;

        const auto& A = anchors_v3[i];
        float a = A[0], b = A[1], c = A[2], d = A[3];

        float w = std::exp(tw) * d;
        float h = std::exp(th) * c;
        float yc = ty * c + a;
        float xc = tx * d + b;

        float ymin = yc - h * 0.5f;
        float xmin = xc - w * 0.5f;
        float ymax = yc + h * 0.5f;
        float xmax = xc + w * 0.5f;

        float ww = std::max(0.f, xmax - xmin);
        float hh = std::max(0.f, ymax - ymin);
        dec.boxes_norm.emplace_back(xmin, ymin, ww, hh);

        std::vector<float> p(num_classes_raw);
        for (int cidx = 0; cidx < num_classes_raw; ++cidx) {
            p[cidx] = sigmoidf(scores_raw[i * num_classes_raw + cidx]);
        }
        dec.probs.push_back(std::move(p));
    }
    return dec;
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

    //std::snprintf(fname, sizeof(fname), "%s%s_%s_%04d.jpg", ERR_DIR.c_str(), prefix.c_str(),(ZXing::ToString(err.type())).c_str(), err_id);
    std::snprintf(fname, sizeof(fname),
              "%s%s_%s_%04d.png", ERR_DIR.c_str(), prefix.c_str(),
              ZXing::ToString(err.type()).c_str(), err_id);

    if (cv::imwrite(fname, img))
        std::cout << "[ERR] image saved : " << fname << '\n';
    else
        std::cerr << "[ERR] could not save : " << fname << '\n';

    ++err_id;
}

int main(int argc, char** argv)
{
    
    std::string model_path = "models/detection-precision-npu-2025-08-05T06-13-14.743Z_channel_ptq_vvip.tflite";
    int cam_index = 3; // adjust if needed
    float conf_thresh = 0.70f;
    float iou_thresh  = 0.20f;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) cam_index = std::stoi(argv[2]);

    // Load model
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
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

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors\n";
        return 1;
    }

    // Input tensor
    const int input_idx = interpreter->inputs()[0];
    TfLiteTensor* in = interpreter->tensor(input_idx);
    if (in->type != kTfLiteFloat32 || in->dims->size != 4) {
        std::cerr << "Expected float32 NHWC input\n";
        return 1;
    }
    int in_h = in->dims->data[1];
    int in_w = in->dims->data[2];
    int in_c = in->dims->data[3];
    if (in_c != 3) {
        std::cerr << "Expected 3-channel input\n";
        return 1;
    }

    // Output tensor(s)
    const int out_idx = interpreter->outputs()[0];
    TfLiteTensor* out = interpreter->tensor(out_idx);
    if (out->type != kTfLiteFloat32) {
        std::cerr << "Expected float32 output\n";
        return 1;
    }

    // Camera
    cv::VideoCapture cap(cam_index, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera " << cam_index << "\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FOCUS, 10);

    cv::Mat frame, resized;
    std::vector<double> fts,cpu; const int AVG=10;
    nb_frames_detected =0, frames_decoding=0;
    while (true) {
        double t0=cv::getTickCount();
        if(!cap.read(frame)||frame.empty()) break;

        // Preprocess: resize and map to [-1, 1]
        cv::resize(frame, resized, cv::Size(in_w, in_h));
        resized.convertTo(resized, CV_32FC3, 2.0/255.0, -1.0);

        // Copy into input tensor (NHWC)
        float* in_ptr = interpreter->typed_input_tensor<float>(0);
        std::memcpy(in_ptr, resized.data, resized.total() * resized.elemSize());

        // Inference
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Invoke failed\n";
            break;
        }

        
        // out shape expected: (1, NUM_ANCHORS, 6) with last dim = [score0, score1, ty, tx, th, tw]
        const TfLiteTensor* o = interpreter->tensor(out_idx);
        if (o->dims->size != 3 || o->dims->data[1] != NUM_ANCHORS || o->dims->data[2] != 6) {
            // If your model actually returns two outputs, you can adapt here by
            // taking scores from output[0] (2034x2) and boxes from output[1] (2034x4).
            std::cerr << "Unexpected output shape. Got (";
            for (int i = 0; i < o->dims->size; ++i) {
                std::cerr << o->dims->data[i] << (i+1<o->dims->size ? "," : "");
            }
            std::cerr << "), expected (1," << NUM_ANCHORS << ",6)\n";
            break;
        }

        const float* out_data = interpreter->typed_output_tensor<float>(0);
        // Build temporary arrays for decoder:
        // scores_raw: [NUM_ANCHORS, 2]
        // boxes_raw:  [NUM_ANCHORS, 4] (ty, tx, th, tw)
        std::vector<float> scores_raw(NUM_ANCHORS * 2);
        std::vector<float> boxes_raw(NUM_ANCHORS * 4);
        for (int i = 0; i < NUM_ANCHORS; ++i) {
            const float* row = out_data + i*6;
            scores_raw[i*2 + 0] = row[0]; // background
            scores_raw[i*2 + 1] = row[1]; // QR_CODE
            boxes_raw[i*4 + 0]  = row[2]; // ty
            boxes_raw[i*4 + 1]  = row[3]; // tx
            boxes_raw[i*4 + 2]  = row[4]; // th
            boxes_raw[i*4 + 3]  = row[5]; // tw
        }

        // Decode to normalized boxes + probs
        Decoded dec = decoder_v3(boxes_raw.data(), scores_raw.data(), NUM_ANCHORS, /*num_classes_raw=*/2);

        // Map normalized boxes ? pixel boxes; remove background & threshold
        const int imgW = frame.cols, imgH = frame.rows;
        std::vector<Detection> dets;
        dets.reserve(NUM_ANCHORS);

        for (int i = 0; i < NUM_ANCHORS; ++i) {
            // classes: [background, QR_CODE]; drop background:
            float bg = dec.probs[i][0];
            float qr = dec.probs[i][1];
            float score = qr; // we only care about QR class
            if (score < conf_thresh) continue;

            const auto& b = dec.boxes_norm[i];
            // convert to pixel coords; clamp
            int x = std::max(0, std::min(imgW - 1, int(b.x * imgW)));
            int y = std::max(0, std::min(imgH - 1, int(b.y * imgH)));
            int w = std::max(0, std::min(imgW - x, int(b.width  * imgW)));
            int h = std::max(0, std::min(imgH - y, int(b.height * imgH)));
            if (w <= 1 || h <= 1) continue;

            dets.push_back(Detection{cv::Rect(x, y, w, h), score, /*class_id=*/0});
        }

        // NMS
        auto kept = nms(dets, iou_thresh);

        // Draw and decode with ZXing
        for (const auto& d : kept) {
            // Crop ROI safely
            cv::Rect roiRect = d.box & cv::Rect(0, 0, imgW, imgH);
            if (roiRect.width < 2 || roiRect.height < 2) continue;

            // Show the ROI
            cv::rectangle(frame, d.box, {0, 255, 255}, 2);
            cv::putText(frame, "QR (" + std::to_string(d.score) + "% )", d.box.tl(),cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);

            cv::Mat roi = frame(roiRect).clone();

            // ZXing: build ImageView over BGR data
            ZXing::ImageView iv(roi.data, roi.cols, roi.rows, ZXing::ImageFormat::BGR);
            ZXing::DecodeHints hints;
            hints.setTryHarder(true).setFormats(ZXing::BarcodeFormat::QRCode | ZXing::BarcodeFormat::Aztec);

            double tbegin=cv::getTickCount();
            auto result = ZXing::ReadBarcode(iv, hints);
            double tend=cv::getTickCount();
            double zx_ms = (tend-tbegin)/cv::getTickFrequency()*1000.0;
            int x = d.box.x;
            int y = d.box.y;
           
            nb_frames_detected = nb_frames_detected + 1;
            printf("nb_frames_detected : %d\n",nb_frames_detected);
            printf("frames_decoding : %d\n",frames_decoding);
            if (nb_frames_detected ==100){
                printf("100 frames detected, ratio : %d\n", (frames_decoding*100)/nb_frames_detected);
                nb_frames_detected = 0;
                frames_decoding = 0;
            }


            if (result.isValid()){
                frames_decoding = frames_decoding + 1;
                auto p=result.position();
                if (!result.text().empty()) {
                    std::vector<cv::Point> pts={
                        {p.topLeft().x +x ,p.topLeft().y +y },
                        {p.topRight().x +x,p.topRight().y +y},
                        {p.bottomRight().x +x,p.bottomRight().y +y},
                        {p.bottomLeft().x +x,p.bottomLeft().y +y}};
                    cv::polylines(frame,pts,true,COLOR_GRN,2);
                    std::cout<<'['<<args_width<<'x'<<args_height<<"] Decoded Text in "<<std::fixed<<std::setprecision(1)<<zx_ms<<" ms using ML : "
                            <<result.text()<<" ("<<ZXing::ToString(result.format())<<")\n";
                    break;
                    }
            }else {
            const ZXing::Error& err = result.error();
            save_error(roi, "roi", err);
            }
                
        }
        fts.push_back((cv::getTickCount()-t0)/cv::getTickFrequency());
        cpu.push_back(cpu_pct());
        if(fts.size()==AVG){
            double fps=AVG/std::accumulate(fts.begin(),fts.end(),0.0);
            double c  =std::accumulate(cpu.begin(),cpu.end(),0.0)/AVG;
            std::cout<<std::fixed<<std::setprecision(1) <<"[INFO] Mean FPS (each "<<AVG<<" frames) : "<<fps<<" | CPU "<<c<<"%\n";
            fts.clear(); cpu.clear();
        }


        cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::flip(frame,frame,1);

        // Target aim overlay
        //cv::putText(frame,"Fit your Qr code in",{frame.cols/2-85,frame.rows/2-10},cv::FONT_HERSHEY_SIMPLEX,0.7,COLOR_AIM,2);
        int cx=frame.cols/2, cy=frame.rows/2, L=40,SZ=170;
        cv::line(frame,{cx-L,cy},{cx+L,cy},COLOR_AIM,2);
        cv::line(frame,{cx,cy-L},{cx,cy+L},COLOR_AIM,2);
        cv::rectangle(frame,{cx-SZ,cy-SZ},{cx+SZ,cy+SZ},COLOR_AIM,2);

        cv::imshow("QR/Barcode Scanner", frame);
        if ((cv::waitKey(1) & 0xFF) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;

}
