#ifndef BOA_WINOGRAD_CONV_H_
#define BOA_WINOGRAD_CONV_H_

#include <vector>

class DirectConv
{
public:
    void operator()(int batch_size, int num_output, int in_channels,
                    const float* input, int input_h, int input_w,
                    const float* weight, int kernel_size,
                    float* output);
};

class GemmConv
{
public:
    void operator()(int batch_size, int num_output, int in_channels,
                    const float* input, int input_h, int input_w,
                    const float* weight, int kernel_size,
                    float* output);

private:
    void im2col(const float* input, int in_channels,
                int input_h, int input_w, int kernel_size,
                int out_h, int out_w, float* cols);
};

class WinogradConv
{
public:
    WinogradConv(int n, int r);
    void operator()(int batch_size, int num_output, int in_channels,
                    const float* input, int input_h, int input_w,
                    const float* weight, int kernel_size,
                    float* output);

private:
    void weight_transform(int num_output,
                          int in_channels,
                          const float* gmat,
                          const float* weight,
                          float* transformed_weight);
    void input_transform(int in_channels,
                         int input_h, int input_w,
                         const float* btmat,
                         const float* input,
                         float* transformed_input);
    void output_transform(const float* atmat,
                          const float* m,
                          float* output);

    inline void matmul(int m, int n, int s, const float* a,
                const float* b, float* c);

    inline void avx_hadamard(const float* a, const float* b,
                             float* c, int length, int channels);

    inline void avx_matmul_line(const float* a, const float* b,
                                float* c, int length);

private:
    int     m_n, m_r, m_alpha;
    std::vector<float>  _g;
    std::vector<float>  _aT;
    std::vector<float>  _bT;

    const int _threads_num {4};
};

class WinogradTransformFactory
{
public:
    static bool getTransform(int n, int r,
                             std::vector<float>& g,
                             std::vector<float>& at,
                             std::vector<float>& bt);
};

void ncnn_conv3x3s1(
    int h, int w, int inch,
    int outw, int outh, int outch,
    const float* bottom_blob,
    float* top_blob,
    const float* kernel);

void conv3x3s1_winograd43_sse(
    int w, int h, int inch,
    int outw, int outh, int outch, 
    const float* bottom_blob,
    float* top_blob,
    const float* kernel);

#endif // BOA_WINOGRAD_CONV_H_