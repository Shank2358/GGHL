#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "buffers.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "common.h"
#include "logger.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "OBBdet.h"
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)
#define DEVICE 0
#define NMS_THRESH 0.45
#define BOX_CONF_THRESH 0.3
#define RUN_FP16 false
#define RUN_INT8 false
#define BATCH_SIZE 1
using namespace cv;
using namespace std;
static const int input_W = 800;
static const int input_H = 800;

// 张量引擎的输入输出程序
const char *INPUT_ENGINE_NAME = "images";
const char *OUTPUT_ENGINE_NAME = "output";
int num_class = 15; // DOTA nun_classes
static const char* labels[]={"plane",
                                "baseball-diamond",
                                "bridge",
                                "ground-track-field",
                                "small-vehicle",
                                "large-vehicle",
                                "ship",
                                "tennis-court",
                                "basketball-court",
                                "storage-tank", "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter"};
const int color_list[15][3] =
    {
        {216, 82, 24},
        {236, 176, 31},
        {125, 46, 141},
        {118, 171, 47},
        {76, 189, 237},
        {238, 19, 46},
        {76, 76, 76},
        {153, 153, 153},
        {255, 0, 0},
        {255, 127, 0},
        {190, 190, 0},
        {0, 255, 0},
        {0, 0, 255},
        {170, 0, 255},
        {84, 84, 0},
};
class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};
cv::Mat static_resize(cv::Mat img)
{
    float resize_ratio = std::min(input_W / (img.cols * 1.0), input_H / (img.rows * 1.0));
    int resize_w = int(resize_ratio * img.cols);
    int resize_h = int(resize_ratio * img.rows);
    cv::Mat re(resize_h, resize_w, CV_8UC3);
    int dw = (input_W - resize_w) / 2;
    int dh = (input_H - resize_h) / 2;
    cv::resize(img, re, re.size());
    cv::Mat out(input_W, input_H, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(dw, dh, re.cols, re.rows))); //修正到对应的部分
    out.convertTo(out, CV_32FC3, 1 / 255.0);            //归一化，转换到对应的0-1均值
    return out;
}
void draw_objects(cv::Mat bgr, vector<OBBInfo8> objects)
{
    cv::Mat img = bgr.clone();
    for (size_t i = 0; i < objects.size(); i++)
    {
        OBBInfo8 obj = objects[i];
        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        char text[256];
        cout << labels[obj.label] << endl;
        sprintf(text, "%s %.1f%%", labels[obj.label], obj.conf * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        cv::Scalar txt_color;
        vector<cv::Point> poly;
        float value_x = (obj.x1 + obj.x2 + obj.x3 + obj.x4) / 4.0;
        float value_y = (obj.y1 + obj.y2 + obj.y3 + obj.y4) / 4.0;
        poly.push_back(cv::Point(obj.x1 + value_x, obj.y1+value_y));
        poly.push_back(cv::Point(obj.x2 + value_x, obj.y2+value_y));
        poly.push_back(cv::Point(obj.x3+ value_x, obj.y3+value_y));
        poly.push_back(cv::Point(obj.x4 + value_x, obj.y4+value_y));
        cv::polylines(img, poly, true, color,4);
        cv::putText(img, text, cv::Point(obj.x1 + value_x, obj.y1 + value_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }
    cv::imwrite("/home/crescent/GGHL/deploy/TensorRT/QAQ858.png", img);
}
class GGHLONNX
{
public:
    GGHLONNX(const string &onnx_file, const string &engine_file) : m_onnx_file(onnx_file), m_engine_file(engine_file){};
    vector<float> prepareImage(const cv::Mat img);
    bool onnxToTRTModel(nvinfer1::IHostMemory *trt_model_stream);
    bool loadEngineFromFile();
    void doInference(const cv::Mat &img);

private:
    const string m_onnx_file;
    const string m_engine_file;
    samplesCommon::Args gArgs;
    nvinfer1::ICudaEngine *m_engine;

    bool constructNetwork(nvinfer1::IBuilder *builder, nvinfer1::INetworkDefinition *network, nvinfer1::IBuilderConfig *config, nvonnxparser::IParser *parser);

    bool saveEngineFile(nvinfer1::IHostMemory *data);
    std::unique_ptr<char[]> readEngineFile(int &length);

    int64_t volume(const nvinfer1::Dims &d);
    unsigned int getElementSize(nvinfer1::DataType t);
};

vector<float> GGHLONNX::prepareImage(const cv::Mat img)
{
    int c = 3;
    float resize_ratio = std::min(input_W / (img.cols * 1.0), input_H / (img.rows * 1.0));
    int resize_w = int(resize_ratio * img.cols);
    int resize_h = int(resize_ratio * img.rows);
    cv::Mat re(resize_h, resize_w, CV_8UC3);
    int dw = (input_W - resize_w) / 2;
    int dh = (input_H - resize_h) / 2;
    cv::resize(img, re, re.size());
    cv::Mat out(input_W, input_H, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(dw, dh, re.cols, re.rows))); //修正到对应的部分
    out.convertTo(out, CV_32FC3, 1.f / 255.0);          //归一化，转换到对应的0-1均值

    // HWC TO CHW
    cout << out.cols << out.rows << endl;
    vector<cv::Mat> input_channels(c);
    cv::split(out, input_channels);

    vector<float> result(input_H * input_W * c);
    auto data = result.data();
    int channel_length = input_H * input_W;
    for (int i = 0; i < c; ++i)
    {
        memcpy(data, input_channels[i].data, channel_length * sizeof(float));
        data += channel_length; // 指针后移channel_length个单位
    }
    return result;
}

bool GGHLONNX::constructNetwork(nvinfer1::IBuilder *builder, nvinfer1::INetworkDefinition *network, nvinfer1::IBuilderConfig *config, nvonnxparser::IParser *parser)
{
    // 解析onnx文件
    if (!parser->parseFromFile(this->m_onnx_file.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity())))
    {
        sample::gLogError << "Fail to parse ONNX file" << std::endl;
        return false;
    }

    // build the Engine
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(1 << 30);
    if (RUN_FP16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (RUN_INT8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

    return true;
}

// 保存plan文件数据
bool GGHLONNX::saveEngineFile(nvinfer1::IHostMemory *data)
{
    std::ofstream file;
    file.open(m_engine_file, std::ios::binary | std::ios::out);
    cout << "writing engine file..." << endl;
    file.write((const char *)data->data(), data->size());
    cout << "save engine file done" << endl;
    file.close();
    return true;
}

// 从plan文件读取数据
std::unique_ptr<char[]> GGHLONNX::readEngineFile(int &length)
{
    ifstream file;
    file.open(m_engine_file, std::ios::in | std::ios::binary);
    // 获得文件流的长度
    file.seekg(0, std::ios::end); // 把指针移到末尾
    length = file.tellg();        // 返回当前指针位置
    // 指针移到开始
    file.seekg(0, std::ios::beg);
    // 定义缓存
    std::unique_ptr<char[]> data(new char[length]);
    // 读取文件到缓存区
    file.read(data.get(), length);
    file.close();
    return data;
}

// 累积乘法 对binding的维度累乘 (3,224,224) => 3*224*224
inline int64_t GGHLONNX::volume(const nvinfer1::Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int GGHLONNX::getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

// 读取plan文件数据，构建engine
bool GGHLONNX::loadEngineFromFile()
{
    int length = 0; // 记录data的长度
    std::unique_ptr<char[]> data = readEngineFile(length);
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
    m_engine = runtime->deserializeCudaEngine(data.get(), length);
    if (!m_engine)
    {
        std::cout << "Failed to create engine" << std::endl;
        return false;
    }
    return true;
}

void GGHLONNX::doInference(const cv::Mat &img)
{
    cv::Mat copy_img = img.clone();
    nvinfer1::IExecutionContext *context = m_engine->createExecutionContext();
    assert(context != nullptr);
    int nbBindings = m_engine->getNbBindings();
    assert(nbBindings == 2); // 输入和输出，一共是2个

    // 为输入和输出创建空间
    void *buffers[2];                 // 待创建的空间  为指针数组
    std::vector<int64_t> buffer_size; // 要创建的空间大小
    buffer_size.resize(nbBindings);
    for (int i = 0; i < nbBindings; i++)
    {
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);    // (3, 224, 224)  (1000)
        nvinfer1::DataType dtype = m_engine->getBindingDataType(i); // 0, 0 也就是两个都是kFloat类型
        // std::cout << static_cast<int>(dtype) << endl;
        int64_t total_size = volume(dims) * 1 * getElementSize(dtype);
        buffer_size[i] = total_size;
        CHECK(cudaMalloc(&buffers[i], total_size));
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream)); // 创建异步cuda流
    auto out_dim = m_engine->getBindingDimensions(1);
    auto output_size = 1;
    for (int j = 0; j < out_dim.nbDims; j++)
    {
        output_size *= out_dim.d[j];
    }
    float *out = new float[output_size];

    // 开始推理
    auto t_start = std::chrono::high_resolution_clock::now();
    vector<float> cur_input = prepareImage(img);
    auto t_end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "loading image takes " << duration << "ms" << std::endl;
    if (!cur_input.data())
    {
        std::cout << "failed to prepare image" << std::endl;
    }

    // 将输入传递到GPU
    CHECK(cudaMemcpyAsync(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice, stream));

    // 异步执行
    t_start = std::chrono::high_resolution_clock::now();
    context->execute(BATCH_SIZE, buffers);
    t_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    // 输出传回给CPU
    CHECK(cudaMemcpyAsync(out, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    
    vector<float> original_result;
    for (int i = 0; i < output_size; i++)
    {
        original_result.push_back(*(out + i));
    }
    cout << output_size << endl;
    vector<OBBInfo> convert_middle = convert_result(original_result);
    cout << convert_middle.size() << endl;
    vector<OBBInfo8> middle = convert_pred(convert_middle, 800, 1024, BOX_CONF_THRESH);
    cv::imwrite("/home/crescent/GGHL/deploy/TensorRT/QAQ86.png", img);
    cout << middle.size() << endl;
    vector<OBBInfo8> result = non_max_supression_8_points(middle, BOX_CONF_THRESH, NMS_THRESH);
    draw_objects(copy_img, result);
    cout << result.size() << endl;
}

/*
 * 在没有trt engine plan文件的情况下，从onnx文件构建engine，然后序列化成engine plan文件
 */
bool GGHLONNX::onnxToTRTModel(nvinfer1::IHostMemory *trt_model_stream)
{
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    assert(builder != nullptr);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());

    // 构建网络
    if (!constructNetwork(builder, network, config, parser))
    {
        return false;
    }
    // 构建引擎
    m_engine = builder->buildEngineWithConfig(*network, *config);
    assert(m_engine != nullptr);
    // 验证网络构建正确
    assert(network->getNbInputs() == 1);


    // 序列化
    trt_model_stream = m_engine->serialize();
    nvinfer1::IHostMemory *data = m_engine->serialize();
    saveEngineFile(data);

    parser->destroy();
    network->destroy();
    builder->destroy();
    // m_engine->destroy();
}

int main(int argc, char **argv)
{
    string input_image_path = "/home/crescent/images/P0006__1024__0___0.png";
    //这边进行一个初始化的过程
    const char *onnxfilename = "/home/crescent/GGHL/GGHL.onnx";
    string engine_file = "/home/crescent/GGHL/GGHL.trt";
    cv::Mat img = cv::imread(input_image_path);
    IHostMemory *trt_model_stream{nullptr};
    GGHLONNX GGHL = GGHLONNX(onnxfilename, engine_file);
    fstream engine_reader;
    engine_reader.open(engine_file, std::ios::in);
    if (engine_reader)
    {
        std::cout << "found engine plan" << endl;
        GGHL.loadEngineFromFile();
    }
    else
    {
        GGHL.onnxToTRTModel(trt_model_stream);
    }
    GGHL.doInference(img);
    /*
    GGHLDet detector(onnxfilename);

    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = static_resize(img);
    float scale = std::min(input_W / (img.cols * 1.0), input_H / (img.rows * 1.0));
    float *blob;
    blob = blobFromImage(pr_img);
    bool demo = detector.build();
    auto out_dims = detector.mEngine->getBindingDimensions(1);
    auto output_size = 1;
    for (int j = 0; j < out_dims.nbDims; j++)
    {
        output_size = output_size * out_dims.d[j];
    }
    static float *prob = new float[output_size];
    cout << output_size << endl;
    detector.infer(blob, prob, output_size);
    vector<float> original_result;
    for (int i = 0; i < output_size; i++)
    {
        original_result.push_back(*(prob + i));
    }
    vector<OBBInfo> convert_middle = convert_result(original_result);
    vector<OBBInfo8> middle = convert_pred(convert_middle, 800, 1024, BOX_CONF_THRESH);
    vector<OBBInfo8> result = non_max_supression_8_points(middle, BOX_CONF_THRESH, NMS_THRESH);
    draw_objects(img,result);
    */
}
