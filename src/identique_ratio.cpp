#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// ZXing-C++ 
#include <ZXing/BarcodeFormat.h>
#include <ZXing/ReadBarcode.h>
#include <ZXing/Result.h>
#include <ZXing/ImageView.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "anchors_v3.h" // generated header with NUM_ANCHORS=2034 and anchors_v3[][]


inline float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }

struct Detection {
    cv::Rect box;   // pixel coords
    float score;    // confidence
    int class_id;   // 0..num_classes-1 (after background removal)
};

// Simple NMS (IoU on pixel rects)
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

// ------------------ Decoder  ------------------
// Input:
//  - boxes_raw:  [num_boxes, 4] in order (ty, tx, th, tw) BEFORE scaling
//  - scores_raw: [num_boxes, num_classes_raw] logits (we will sigmoid)
// Anchors are (a, b, c, d) in anchors_v3[i] (normalized)
// Output:
//  - boxes_norm: Rects in normalized coords [0..1] (x,y,w,h)
//  - probs:      per-box probabilities (sigmoid), with background kept or dropped by caller
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
        // Apply the same scaling as Python: ty/tx /10, th/tw /5
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

// ------------------ App ------------------

int main(int argc, char** argv)
{
    std::string model_path = "detection-precision-npu-2025-08-05T06-13-14.743Z_channel_ptq_vvip.tflite";
    int cam_index = 4; // adjust if needed
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
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk || !interpreter) {
        std::cerr << "Failed to create interpreter\n";
        return 1;
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
    // Your Python split suggests ONE output with shape (1, 2034, 6) ? 2 scores + 4 box params.
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
    std::cout << "Ready. Press 'q' to quit.\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

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
            cv::rectangle(frame, d.box, {0, 255, 0}, 2);
            cv::putText(frame, "QR (" + std::to_string(d.score) + ")", d.box.tl(),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);

            // Crop ROI safely
            cv::Rect roiRect = d.box & cv::Rect(0, 0, imgW, imgH);
            if (roiRect.width < 2 || roiRect.height < 2) continue;

            cv::Mat roi = frame(roiRect).clone();

            // ZXing: build ImageView over BGR data
            ZXing::ImageView iv(roi.data, roi.cols, roi.rows, ZXing::ImageFormat::BGR);
            ZXing::DecodeHints hints;
            hints.setTryHarder(true).setFormats(ZXing::BarcodeFormat::QRCode | ZXing::BarcodeFormat::Aztec);

            auto result = ZXing::ReadBarcode(iv, hints);
            if (result.isValid()) {
                std::string text = result.text();
                std::cout << "Decoded: " << text << "\n";
                cv::putText(frame, text, d.box.br() + cv::Point(-d.box.width, -5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
            }
        }

        cv::imshow("QR/Barcode Scanner", frame);
        if ((cv::waitKey(1) & 0xFF) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
