#include "conv.h"
#include <assert.h>
#include <omp.h>
#include <immintrin.h>
#include <iostream>

using namespace std;

void DirectConv::operator()(
    int batch_size, int num_output, int in_channels,
    const float* input, int input_h, int input_w,
    const float* weight, int kernel_size, float* output)
{
    int out_h = input_h - kernel_size + 1;
    int out_w = input_w - kernel_size + 1;

    for (int i = 0; i < batch_size; i++)
    {
        const float* ptr_in_batch = input + i * in_channels * input_h * input_w;
        float* ptr_out_batch = output + i * num_output * out_h * out_w;
        
        for (int j = 0; j < num_output; j++)
        {
            const float* ptr_weight_out = weight + j * in_channels * kernel_size * kernel_size;
            float* ptr_out_channel = ptr_out_batch + j * out_h * out_w;

            for (int y = 0; y < out_h; y++)
            {
                float* ptr_out = ptr_out_channel + y * out_w;
                for (int x = 0; x < out_w; x++)
                {
                    const float* ptr_in_x = ptr_in_batch + y * input_w + x;
                    float val = 0.f;
                    for (int c = 0; c < in_channels; c++)
                    {
                        const float* ptr_weight_channel = ptr_weight_out + c * kernel_size * kernel_size;
                        const float* ptr_in_channel = ptr_in_x + c * input_h * input_w;
                        for (int kh = 0; kh < kernel_size; kh++)
                        {
                            const float* ptr_weight = ptr_weight_channel + kh * kernel_size;
                            const float* ptr_in = ptr_in_channel + kh * input_w;

                            for (int kw = 0; kw < kernel_size; kw++)
                            {
                                val += ptr_in[kw] * ptr_weight[kw];
                            }
                        }
                    }
                    ptr_out[x] = val;
                }
            }
        }
    }
}

void GemmConv::operator()(
    int batch_size, int num_output, int in_channels,
    const float* input, int input_h, int input_w,
    const float* weight, int kernel_size, float* output)
{
    int out_h = input_h - kernel_size + 1;
    int out_w = input_w - kernel_size + 1;
    int kernel_cube_size = in_channels * kernel_size * kernel_size;
    int cols_cols = out_h * out_w;
    float* cols = new float[cols_cols * kernel_cube_size];

    for (int i = 0; i < batch_size; i++)
    {
        const float* ptr_in_batch = input + i * in_channels * input_h * input_w;
        float* ptr_out_batch = output + i * num_output * out_h * out_w;

        im2col(ptr_in_batch, in_channels, input_h, input_w,
               kernel_size, out_h, out_w, cols);

        for (int j = 0; j < num_output; j++)
        {
            const float* ptr_weight = weight + j * kernel_cube_size;
            float* ptr_out = ptr_out_batch + j * cols_cols;
            
            for (int k = 0; k < cols_cols; k++)
            {
                const float* ptr_cols = cols + k * kernel_cube_size;
                float val = 0.f;
                for (int n = 0; n < kernel_cube_size; n++)
                {
                    val += ptr_cols[n] * ptr_weight[n];
                }
                ptr_out[k] = val;
            }
        }
    }
}

void GemmConv::im2col(
    const float* input, int in_channels,
    int input_h, int input_w, int kernel_size,
    int out_h, int out_w, float* cols)
{
    int col_mat_rows = in_channels * 
                       kernel_size * kernel_size;
    int* offset = new int[col_mat_rows];

    for (int ch = 0; ch < in_channels; ch++)
    {
        int spatial_offset = ch * input_h * input_w;
        int* ptr_spatial_offset = offset + ch * kernel_size * kernel_size;
        for (int i = 0; i < kernel_size; i++)
        {
            int row_offset = spatial_offset + i * input_w;
            int* ptr_row_offset = ptr_spatial_offset + i * kernel_size;
            for (int j = 0; j < kernel_size; j++)
            {
                ptr_row_offset[j] = row_offset + j;
            }
        }
    }
    
    for (int i = 0; i < out_h; i++)
    {
        for (int j = 0; j < out_w; j++)
        {
            float* ptr_cols = cols + (i * out_w + j) * col_mat_rows;
            const float* ptr_in = input + i * input_w + j;
            for (int n = 0; n < col_mat_rows; n++)
            {
                ptr_cols[n] = ptr_in[offset[n]];
            }
        }
    }
}

WinogradConv::WinogradConv(int n, int r)
    : m_n(n), m_r(r), m_alpha(n+r-1)
{
    WinogradTransformFactory::getTransform(
        n, r, _g, _aT, _bT
    );
}

void WinogradConv::operator()(
    int batch_size, int num_output, int in_channels,
    const float* input, int input_h, int input_w,
    const float* weight, int kernel_size, float* output)
{
    // 需要在调用前padding
    assert(kernel_size == m_r);
    int out_h = input_h - kernel_size + 1;
    int out_w = input_w - kernel_size + 1;
    assert(out_h % m_n == 0);
    assert(out_w % m_n == 0);

    // 中间结果缓存
    int spatial_size = m_alpha * m_alpha;
    float* GgTGT = new float[num_output * in_channels * spatial_size];
    // float* BTdTB = new float[in_channels * spatial_size];
    // float* m_buffer = new float[spatial_size];
    // float* temp = new float[m_n * m_n];

    // GgTGT
    weight_transform(num_output, in_channels, _g.data(), weight, GgTGT);

    // elementwise multiply and sum
    for (int i = 0; i < batch_size; i++)
    {
        const float* ptr_in_batch = input + i * in_channels * input_h * input_w;
        float* ptr_out_batch = output + i * num_output * out_h * out_w;

        #pragma omp parallel for num_threads(4)
        for (int y = 0; y < out_h; y += m_n)
        {
            // #pragma omp parallel for num_threads(4)
            for (int x = 0; x < out_w; x += m_n)
            {
                float* BTdTB = new float[in_channels * spatial_size];
                float* m_buffer = new float[spatial_size];
                float* temp = new float[m_n * m_n];

                // BTdTB
                const float* ptr_in = ptr_in_batch + y * input_w + x;
                input_transform(in_channels, input_h, input_w,
                                _bT.data(), ptr_in, BTdTB);

                float* ptr_out_spatial = ptr_out_batch + y * out_w + x;

                for (int j = 0; j < num_output; j++)
                {
                    const float* ptr_weight = GgTGT + j * in_channels * spatial_size;
                    float* ptr_out = ptr_out_spatial + j * out_h * out_w;
                    
                    // hadamard
                    // for (int a = 0; a < spatial_size; a++)
                    // {
                    //     m_buffer[a] = ptr_weight[a] * BTdTB[a];
                    // }
                    avx_hadamard(ptr_weight, BTdTB, m_buffer, spatial_size, in_channels);

                    // for (int ch = 1; ch < in_channels; ch++)
                    // {
                    //     const float* ptr0 = ptr_weight + ch * spatial_size;
                    //     const float* ptr1 = BTdTB + ch * spatial_size;

                    //     for (int a = 0; a < spatial_size; a++)
                    //     {
                    //         m_buffer[a] += ptr0[a] * ptr1[a];
                    //     }
                    // }

                    // AMTAT
                    output_transform(_aT.data(), m_buffer, temp);
                    for (int ceil_y = 0; ceil_y < m_n; ceil_y++)
                    {
                        float* ptr = ptr_out + ceil_y * out_w;
                        for (int ceil_x = 0; ceil_x < m_n; ceil_x++)
                        {
                            ptr[ceil_x] = temp[ceil_y * m_n + ceil_x];
                        }
                    }
                }

                delete[] BTdTB;
                delete[] m_buffer;
                delete[] temp;
            }
        }
    }

    delete[] GgTGT;
    // delete[] BTdTB;
    // delete[] m_buffer;
    // delete[] temp;
}

void WinogradConv::weight_transform(
    int num_output, int in_channels,
    const float* gmat,
    const float* weight, float* transformed_weight)
{
    float* temp = new float[m_alpha * m_r];
    float* g_tranpose = new float[m_r * m_r];

    for (int i = 0; i < num_output; i++)
    {
        const float* ptr_weight_out = weight + i * in_channels * m_r * m_r;
        float* ptr_out = transformed_weight + i * in_channels * m_alpha * m_alpha;

        for (int j = 0; j < in_channels; j++)
        {
            const float* ptr_weight_in = ptr_weight_out + j * m_r * m_r;
            float* ptr = ptr_out + j * m_alpha * m_alpha;

            // g_transpose
            for (int y = 0; y < m_r; y++)
            {
                const float* ptr_weight = ptr_weight_in + y * m_r;
                float* ptr_g = g_tranpose + y * m_r;

                for (int x = 0; x < m_r; x++)
                {
                    ptr_g[x] = ptr_weight[x];
                }
            }
            
            // GgT
            matmul(m_alpha, m_r, m_r, gmat, g_tranpose, temp);
            
            // GgTGT
            matmul(m_alpha, m_alpha, m_r, temp, gmat, ptr);
        }
    }

    delete[] temp;
    delete[] g_tranpose;
}

void WinogradConv::input_transform(
    int in_channels, int input_h, int input_w,
    const float* btmat,
    const float* input, float* transformed_input)
{
    float* temp = new float[m_alpha * m_alpha];
    
    for (int ch = 0; ch < in_channels; ch++)
    {
        const float *ptr_in_channel = input + ch * input_h * input_w;
        float* ptr_output = transformed_input + ch * m_alpha * m_alpha;

        // roi
        for (int y = 0; y < m_alpha; y++)
        {
            const float *ptr_in = ptr_in_channel + y * input_w;
            float* ptr = ptr_output + y * m_alpha;

            for (int x = 0; x < m_alpha; x++)
            {
                ptr[x] = ptr_in[x];
            }
        }
        
        // BTdT
        matmul(m_alpha, m_alpha, m_alpha, btmat, ptr_output, temp);

        // BTdTB
        matmul(m_alpha, m_alpha, m_alpha, temp, btmat, ptr_output);
    }
    
    delete[] temp;
}

void WinogradConv::output_transform(
    const float* atmat, const float* m,
    float* output)
{
    float* buffer = new float[m_n * m_alpha];

    // ATm
    matmul(m_n, m_alpha, m_alpha, atmat, m, buffer);
    // ATmA
    matmul(m_n, m_n, m_alpha, buffer, atmat, output);

    delete[] buffer;
}

void WinogradConv::matmul(int m, int n, int s,
    const float* a, const float* b, float* c)
{
    // suppose b is already transposed
    for (int y = 0; y < m; y++)
    {
        const float *ptr_a = a + y * s;
        float *ptr = c + y * n;

        for (int x = 0; x < n; x++)
        {
            const float *ptr_b = b + x * s;

            // avx_matmul_line(ptr_a, ptr_b, ptr+x, s);

            float val = 0.f;
            for (int i = 0; i < s; i++)
            {
                val += ptr_a[i] * ptr_b[i];
            }

            ptr[x] = val;
        }   
    }  
}

void WinogradConv::avx_hadamard(const float* a,
                                const float* b,
                                float* c,
                                int length,
                                int channel)
{
    // avx2
    int nn = length >> 3;
    int remain = length - (nn << 3);

    int n = nn;
    float* ptr = c;

    while (n > 0)
    {
        __m256 m_a = _mm256_loadu_ps(a);
        __m256 m_b = _mm256_loadu_ps(b);
        __m256 m_c = _mm256_mul_ps(m_a, m_b);
        _mm256_storeu_ps(ptr, m_c);

        n--;
        a += 8;
        b += 8;
        ptr += 8;
    }

    for (int i = 0; i < remain; i++)
    {
        ptr[i] = a[i] * b[i];
    }

    a += remain;
    b += remain;
    
    while (--channel)
    {
        n = nn;
        ptr = c;

        while (n > 0)
        {
            __m256 m_a = _mm256_loadu_ps(a);
            __m256 m_b = _mm256_loadu_ps(b);
            __m256 m_c = _mm256_loadu_ps(ptr);

            m_a = _mm256_mul_ps(m_a, m_b);
            m_c = _mm256_add_ps(m_c, m_a);
            _mm256_storeu_ps(ptr, m_c);

            n--;
            a += 8;
            b += 8;
            ptr += 8;
        }

        for (int i = 0; i < remain; i++)
        {
            ptr[i] += a[i] * b[i];
        }

        a += remain;
        b += remain;
    }
}

void WinogradConv::avx_matmul_line(const float* a,
                                   const float* b,
                                   float* c, int length)
{
    // sse
    int nn = length >> 2;
    int remain = length - (nn << 2);

    __m128 m_c = _mm_setzero_ps();
    __m128 m_0 = m_c;

    while (nn > 0)
    {
        __m128 m_a = _mm_loadu_ps(a);
        __m128 m_b = _mm_loadu_ps(b);

        m_a = _mm_mul_ps(m_a, m_b);
        m_c = _mm_add_ps(m_c, m_a);

        nn--;
        a += 4;
        b += 4;
    }

    __m128 m_temp = _mm_unpackhi_ps(m_c, m_0);
    m_c = _mm_unpacklo_ps(m_c, m_0);
    m_c = _mm_add_ps(m_c, m_temp);
    m_temp = _mm_unpackhi_ps(m_c, m_0);
    m_c = _mm_add_ps(m_c, m_temp);
    float val = _mm_cvtss_f32(m_c);

    while (remain)
    {
        val += *a++ * *b++;
        remain--;
    }

    *c = val;
}

bool WinogradTransformFactory::getTransform(
        int n, int r, std::vector<float>& g,
        std::vector<float>& at, std::vector<float>& bt)
{
    if (r == 3 && n == 2)
    {
        g = {1.f,    0.f,  0.f,
             0.5f,  0.5f, 0.5f,
             0.5f, -0.5f, 0.5f,
             0.f,    0.f,  1.f};

        at = {1.f, 1.f, 1.f, 0.f,
              0.f, 1.f, -1.f, 1.f};

        bt = {1.f, 0.f, -1.f, 0.f,
              0.f, 1.f, 1.f, 0.f,
              0.f, -1.f, 1.f, 0.f,
              0.f, -1.f, 0.f, 1.f};
    }
    else if (r == 3 && n == 4)
    {
        g = {1.f/4, 0.f, 0.f,
             -1.f/6, -1.f/6, -1.f/6,
             -1.f/6, 1.f/6, -1.f/6,
             1.f/24, 1.f/12, 1.f/6,
             1.f/24, -1.f/12, 1.f/6,
             0.f, 0.f, 1.f};

        at = {1.f, 1.f, 1.f, 1.f, 1.f, 0.f,
              0.f, 1.f, -1.f, 2.f, -2.f, 0.f,
              0.f, 1.f, 1.f, 4.f, 4.f, 0.f,
              0.f, 1.f, -1.f, 8.f, -8.f, 1.f};

        bt = {4.f, 0.f, -5.f, 0.f, 1.f, 0.f,
              0.f, -4.f, -4.f, 1.f, 1.f, 0.f,
              0.f, 4.f, -4.f, -1.f, 1.f, 0.f,
              0.f, -2.f, -1.f, 2.f, 1.f, 0.f,
              0.f, 2.f, -1.f, -2.f, 1.f, 0.f,
              0.f, 4.f, 0.f, -5.f, 0.f, 1.f};
    }
    else if (r == 3 && n == 6)
    {
        g = {1.f/36, 0.f, 0.f,
             1.f/48, 1.f/48, 1.f/48,
             1.f/48, -1.f/48, 1.f/48,
             -1.f/120, -1.f/60, -1.f/30,
             -1.f/120, 1.f/60, -1.f/30,
             1.f/720, 1.f/240, 1.f/80,
             1.f/720, -1.f/240, 1.f/80,
             0.f, 0.f, 1.f};

        at = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f,
              0.f, 1.f, -1.f, 2.f, -2.f, 3.f, -3.f, 0.f,
              0.f, 1.f, 1.f, 4.f, 4.f, 9.f, 9.f, 0.f,
              0.f, 1.f, -1.f, 8.f, -8.f, 27.f, -27.f, 0.f,
              0.f, 1.f, 1.f, 16.f, 16.f, 81.f, 81.f, 0.f,
              0.f, 1.f, -1.f, 32.f, -32.f, 243.f, -243.f, 1.f};

        bt = {36.f, 0.f, -49.f, 0.f, 14.f, 0.f, -1.f, 0.f,
              0.f, 36.f, 36.f, -13.f, -13.f, 1.f, 1.f, 0.f,
              0.f, -36.f, 36.f, 13.f, -13.f, -1.f, 1.f, 0.f,
              0.f, 18.f, 9.f, -20.f, -10.f, 2.f, 1.f, 0.f,
              0.f, -18.f, 9.f, 20.f, -10.f, -2.f, 1.f, 0.f,
              0.f, 12.f, 4.f, -15.f, -5.f, 3.f, 1.f, 0.f,
              0.f, -12.f, 4.f, 15.f, -5.f, -3.f, 1.f, 0.f,
              0.f, -36.f, 0.f, 49.f, 0.f, -14.f, 0.f, 1.f};
    }
    else if (r == 4 && n == 2)
    {
        g = {1.f/6, 1.f/6, 1.f/6, 1.f/6,
             1.f/6, -1.f/6, 1.f/6, -1.f/6,
             1.f/12, 1.f/6, 1.f/3, 2.f/3,
             -1.f/12, 1.f/6, -1.f/3, 2.f/3,
             0.f, 0.f, 0.f, 1.f};


        at = {1.f, 1.f,
              1.f, -1.f,
              1.f, 2.f,
              1.f, -2.f,
              0.f, 1.f};

        bt = {4.f, 4.f, -2.f, 2.f, 4.f,
              4.f, -4.f, -1.f, -1.f, 0.f,
              -1.f, -1.f, 2.f, -2.f, -5.f,
              -1.f, 1.f, 1.f, 1.f, 0.f,
              0.f, 0.f, 0.f, 0.f, 1.f};
    }
    else if (r == 4 && n == 3)
    {
        g = {1.f/4, 0.f, 0.f, 0.f,
             -1.f/6, -1.f/6, -1.f/6, -1.f/6,
             -1.f/6, 1.f/6, -1.f/6, 1.f/6,
             1.f/24, 1.f/12, 1.f/6, 1.f/3,
             1.f/24, -1.f/12, 1.f/6, -1.f/3,
             0.f, 0.f, 0.f, 1.f};


        at = {1.f, 0.f, 0.f,
              1.f, 1.f, 1.f,
              1.f, -1.f, 1.f,
              1.f, 2.f, 4.f,
              1.f, -2.f, 4.f,
              0.f, 0.f, 1.f};

        bt = {4.f, 0.f, 0.f, 0.f, 0.f, 0.f,
              0.f, -4.f, 4.f, -2.f, 2.f, 4.f,
              -5.f, -4.f, -4.f, -1.f, -1.f, 0.f,
              0.f, 1.f, -1.f, 2.f, -2.f, -5.f,
              1.f, 1.f, 1.f, 1.f, 1.f, 0.f,
              0.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    }
    else if (r == 4 && n == 5)
    {
        g = {1.f/36, 0.f, 0.f, 0.f,
             1.f/48, 1.f/48, 1.f/48, 1.f/48,
             1.f/48, -1.f/48, 1.f/48, -1.f/48,
             -1.f/120, -1.f/60, -1.f/30, -1.f/15,
             -1.f/120, 1.f/60, -1.f/30, 1.f/15,
             1.f/720, 1.f/240, 1.f/80, 3.f/80,
             1.f/720, -1.f/240, 1.f/80, -3.f/80,
             0.f, 0.f, 0.f, 1.f};

        at = {1.f, 0.f, 0.f, 0.f, 0.f,
              1.f, 1.f, 1.f, 1.f, 1.f,
              1.f, -1.f, 1.f, -1.f, 1.f,
              1.f, 2.f, 4.f, 8.f, 16.f,
              1.f, -2.f, 4.f, -8.f, 16.f,
              1.f, 3.f, 9.f, 27.f, 81.f,
              1.f, -3.f, 9.f, -27.f, 81.f,
              0.f, 0.f, 0.f, 0.f, 1.f};

        bt = {36.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
              0.f, 36.f, -36.f, 18.f, -18.f, 12.f, -12.f, -36.f,
              -49.f, 36.f, 36.f, 9.f, 9.f, 4.f, 4.f, 0.f,
              0.f, -13.f, 13.f, -20.f, 20.f, -15.f, 15.f, 49.f,
              14.f, -13.f, -13.f, -10.f, -10.f, -5.f, -5.f, 0.f,
              0.f, 1.f, -1.f, 2.f, -2.f, 3.f, -3.f, -14.f,
              -1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f,
              0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f};
    }
}

void ncnn_conv3x3s1(int h, int w, int inch, int outw, int outh, int outch,
                        const float* bottom_blob, float* top_blob, const float* kernel)
{
    #pragma omp parallel for num_threads(4)
    for (int p = 0; p < outch; p++)
    {
        float* out = top_blob + p * outh * outw;

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = bottom_blob + q * h * w;
            const float* kernel0 = kernel + p * inch * 9 + q * 9;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            int i = 0;

            for (; i + 1 < outh; i += 2)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    float sum = 0;
                    float sum2 = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr += sum;
                    *outptr2 += sum2;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;

                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }
        }
    }
}

void conv3x3s1_winograd43_sse(
    int w, int h, int inch,
    int outw, int outh, int outch, 
    const float* bottom_blob,
    float* top_blob,
    const float* kernel)
{
    // kernel transform GgGT
    float* kernel_tm = new float[36 * inch * outch];

    // G
    const float ktm[6][3] = {
        {1.0f / 4, 0.0f, 0.0f},
        {-1.0f / 6, -1.0f / 6, -1.0f / 6},
        {-1.0f / 6, 1.0f / 6, -1.0f / 6},
        {1.0f / 24, 1.0f / 12, 1.0f / 6},
        {1.0f / 24, -1.0f / 12, 1.0f / 6},
        {0.0f, 0.0f, 1.0f}
    };

    const int threads_num = 4;
    #pragma omp parallel for num_threads(threads_num)
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + (p * inch + q) * 9;
            float* kernel_tm0 = kernel_tm + (p * inch + q) * 36;

            // transform kernel
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[6][3];
            for (int i = 0; i < 6; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 6; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 6; i++)
                {
                    kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // 9 * outch/8 * inch * 32
    float* kernel_tm2[9];
    for (int r = 0; r < 9; r++)
    {
        float* kernel_tm_test = new float[4 * 8 * inch * (outch / 8 + (outch % 8) / 4 + outch % 4)];

        int p = 0;
        for (; p + 7 < outch; p += 8)
        {
            const float* kernel0 = kernel_tm + (p + 0) * inch * 36 + r * 4;
            const float* kernel1 = kernel_tm + (p + 1) * inch * 36 + r * 4;
            const float* kernel2 = kernel_tm + (p + 2) * inch * 36 + r * 4;
            const float* kernel3 = kernel_tm + (p + 3) * inch * 36 + r * 4;
            const float* kernel4 = kernel_tm + (p + 4) * inch * 36 + r * 4;
            const float* kernel5 = kernel_tm + (p + 5) * inch * 36 + r * 4;
            const float* kernel6 = kernel_tm + (p + 6) * inch * 36 + r * 4;
            const float* kernel7 = kernel_tm + (p + 7) * inch * 36 + r * 4;

            float* ktmp = kernel_tm_test + (p / 8) * inch * 32;
            for (int q = 0; q < inch; q++)
            {
                __m128 t = _mm_loadu_ps(kernel0);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                t = _mm_loadu_ps(kernel1);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;
                
                t = _mm_loadu_ps(kernel2);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;
                
                t = _mm_loadu_ps(kernel3);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                t = _mm_loadu_ps(kernel4);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                t = _mm_loadu_ps(kernel5);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                t = _mm_loadu_ps(kernel6);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                t = _mm_loadu_ps(kernel7);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                kernel0 += 36;
                kernel1 += 36;
                kernel2 += 36;
                kernel3 += 36;
                kernel4 += 36;
                kernel5 += 36;
                kernel6 += 36;
                kernel7 += 36;
            }
        }

        for (; p + 3 < outch; p += 4)
        {
            const float* kernel0 = kernel_tm + (p + 0) * inch * 36 + r * 4;
            const float* kernel1 = kernel_tm + (p + 1) * inch * 36 + r * 4;
            const float* kernel2 = kernel_tm + (p + 2) * inch * 36 + r * 4;
            const float* kernel3 = kernel_tm + (p + 3) * inch * 36 + r * 4;

            float* ktmp = kernel_tm_test + (p / 8 + (p % 8) / 4) * inch * 32;

            for (int q = 0; q < inch; q++)
            {
                __m128 t = _mm_loadu_ps(kernel0);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                t = _mm_loadu_ps(kernel1);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;
                
                t = _mm_loadu_ps(kernel2);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                t = _mm_loadu_ps(kernel3);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                kernel0 += 36;
                kernel1 += 36;
                kernel2 += 36;
                kernel3 += 36;
            }
        }

        for (; p < outch; p++)
        {
            const float* kernel0 = kernel_tm + p * inch * 36 + r * 4;
            float* ktmp = kernel_tm_test + (p / 8 + (p % 8) / 4 + p % 4) * inch * 32;

            for (int q = 0; q < inch; q++)
            {
                __m128 t = _mm_loadu_ps(kernel0);
                _mm_storeu_ps(ktmp, t);
                ktmp += 4;

                kernel0 += 36;
            }
        }
        kernel_tm2[r] = kernel_tm_test;
    }

    // tiles
    const int colBlocks = outh / 4;
    const int rowBlocks = outw / 4;
    const int tiles = colBlocks * rowBlocks;

    // BEGIN transform input
    float* bottom_blob_tm = new float[tiles * inch * 36];
    {
        // BT
        // const float itm[6][6] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
        //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
        //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
        // };

        // 0 =	4 * r00  - 5 * r02	+ r04
        // 1 = -4 * (r01 + r02)  + r03 + r04
        // 2 =	4 * (r01 - r02)  - r03 + r04
        // 3 = -2 * r01 - r02 + 2 * r03 + r04
        // 4 =	2 * r01 - r02 - 2 * r03 + r04
        // 5 =	4 * r01 - 5 * r03 + r05

        __m256 _1_n = _mm256_set1_ps(-1);
        __m256 _2_p = _mm256_set1_ps(2);
        __m256 _2_n = _mm256_set1_ps(-2);
        __m256 _4_p = _mm256_set1_ps(4);
        __m256 _4_n = _mm256_set1_ps(-4);
        __m256 _5_n = _mm256_set1_ps(-5);

        #pragma omp parallel for num_threads(threads_num)
        for (int q = 0; q < inch; q++)
        {
            const float* img = bottom_blob + q * h * w;
            for (int j = 0; j < colBlocks; j++)
            {
                const float* r0 = img + j * w * 4;
                const float* r1 = r0 + w;
                const float* r2 = r1 + w;
                const float* r3 = r2 + w;
                const float* r4 = r3 + w;
                const float* r5 = r4 + w;

                for (int i = 0; i < rowBlocks; i++)
                {
                    float* out_tm0 = bottom_blob_tm + (tiles * 0 + j * rowBlocks + i) * inch * 4 + q * 4;
                    float* out_tm1 = bottom_blob_tm + (tiles * 1 + j * rowBlocks + i) * inch * 4 + q * 4;
                    float* out_tm2 = bottom_blob_tm + (tiles * 2 + j * rowBlocks + i) * inch * 4 + q * 4;
                    float* out_tm3 = bottom_blob_tm + (tiles * 3 + j * rowBlocks + i) * inch * 4 + q * 4;
                    float* out_tm4 = bottom_blob_tm + (tiles * 4 + j * rowBlocks + i) * inch * 4 + q * 4;
                    float* out_tm5 = bottom_blob_tm + (tiles * 5 + j * rowBlocks + i) * inch * 4 + q * 4;
                    float* out_tm6 = bottom_blob_tm + (tiles * 6 + j * rowBlocks + i) * inch * 4 + q * 4;
                    float* out_tm7 = bottom_blob_tm + (tiles * 7 + j * rowBlocks + i) * inch * 4 + q * 4;
                    float* out_tm8 = bottom_blob_tm + (tiles * 8 + j * rowBlocks + i) * inch * 4 + q * 4;

                    __m256 _d0, _d1, _d2, _d3, _d4, _d5;
                    __m256 _w0, _w1, _w2, _w3, _w4, _w5;
                    __m256 _t0, _t1, _t2, _t3, _t4, _t5;
                    __m256 _n0, _n1, _n2, _n3, _n4, _n5;

                    // load
                    _d0 = _mm256_loadu_ps(r0);
                    _d1 = _mm256_loadu_ps(r1);
                    _d2 = _mm256_loadu_ps(r2);
                    _d3 = _mm256_loadu_ps(r3);
                    _d4 = _mm256_loadu_ps(r4);
                    _d5 = _mm256_loadu_ps(r5);

                    // w = B_t * d
                    _w0 = _mm256_mul_ps(_d0, _4_p);
                    _w0 = _mm256_fmadd_ps(_d2, _5_n, _w0);
                    _w0 = _mm256_add_ps(_w0, _d4);

                    _w1 = _mm256_mul_ps(_d1, _4_n);
                    _w1 = _mm256_fmadd_ps(_d2, _4_n, _w1);
                    _w1 = _mm256_add_ps(_w1, _d3);
                    _w1 = _mm256_add_ps(_w1, _d4);

                    _w2 = _mm256_mul_ps(_d1, _4_p);
                    _w2 = _mm256_fmadd_ps(_d2, _4_n, _w2);
                    _w2 = _mm256_fmadd_ps(_d3, _1_n, _w2);
                    _w2 = _mm256_add_ps(_w2, _d4);

                    _w3 = _mm256_mul_ps(_d1, _2_n);
                    _w3 = _mm256_fmadd_ps(_d2, _1_n, _w3);
                    _w3 = _mm256_fmadd_ps(_d3, _2_p, _w3);
                    _w3 = _mm256_add_ps(_w3, _d4);

                    _w4 = _mm256_mul_ps(_d1, _2_p);
                    _w4 = _mm256_fmadd_ps(_d2, _1_n, _w4);
                    _w4 = _mm256_fmadd_ps(_d3, _2_n, _w4);
                    _w4 = _mm256_add_ps(_w4, _d4);

                    _w5 = _mm256_mul_ps(_d1, _4_p);
                    _w5 = _mm256_fmadd_ps(_d3, _5_n, _w5);
                    _w5 = _mm256_add_ps(_w5, _d5);

                    // transpose d to d_t
                    {
                        _t0[0] = _w0[0];
                        _t1[0] = _w0[1];
                        _t2[0] = _w0[2];
                        _t3[0] = _w0[3];
                        _t4[0] = _w0[4];
                        _t5[0] = _w0[5];
                        _t0[1] = _w1[0];
                        _t1[1] = _w1[1];
                        _t2[1] = _w1[2];
                        _t3[1] = _w1[3];
                        _t4[1] = _w1[4];
                        _t5[1] = _w1[5];
                        _t0[2] = _w2[0];
                        _t1[2] = _w2[1];
                        _t2[2] = _w2[2];
                        _t3[2] = _w2[3];
                        _t4[2] = _w2[4];
                        _t5[2] = _w2[5];
                        _t0[3] = _w3[0];
                        _t1[3] = _w3[1];
                        _t2[3] = _w3[2];
                        _t3[3] = _w3[3];
                        _t4[3] = _w3[4];
                        _t5[3] = _w3[5];
                        _t0[4] = _w4[0];
                        _t1[4] = _w4[1];
                        _t2[4] = _w4[2];
                        _t3[4] = _w4[3];
                        _t4[4] = _w4[4];
                        _t5[4] = _w4[5];
                        _t0[5] = _w5[0];
                        _t1[5] = _w5[1];
                        _t2[5] = _w5[2];
                        _t3[5] = _w5[3];
                        _t4[5] = _w5[4];
                        _t5[5] = _w5[5];
                    }

                    // d = B_t * d_t
                    _n0 = _mm256_mul_ps(_t0, _4_p);
                    _n0 = _mm256_fmadd_ps(_t2, _5_n, _n0);
                    _n0 = _mm256_add_ps(_n0, _t4);

                    _n1 = _mm256_mul_ps(_t1, _4_n);
                    _n1 = _mm256_fmadd_ps(_t2, _4_n, _n1);
                    _n1 = _mm256_add_ps(_n1, _t3);
                    _n1 = _mm256_add_ps(_n1, _t4);

                    _n2 = _mm256_mul_ps(_t1, _4_p);
                    _n2 = _mm256_fmadd_ps(_t2, _4_n, _n2);
                    _n2 = _mm256_fmadd_ps(_t3, _1_n, _n2);
                    _n2 = _mm256_add_ps(_n2, _t4);

                    _n3 = _mm256_mul_ps(_t1, _2_n);
                    _n3 = _mm256_fmadd_ps(_t2, _1_n, _n3);
                    _n3 = _mm256_fmadd_ps(_t3, _2_p, _n3);
                    _n3 = _mm256_add_ps(_n3, _t4);

                    _n4 = _mm256_mul_ps(_t1, _2_p);
                    _n4 = _mm256_fmadd_ps(_t2, _1_n, _n4);
                    _n4 = _mm256_fmadd_ps(_t3, _2_n, _n4);
                    _n4 = _mm256_add_ps(_n4, _t4);

                    _n5 = _mm256_mul_ps(_t1, _4_p);
                    _n5 = _mm256_fmadd_ps(_t3, _5_n, _n5);
                    _n5 = _mm256_add_ps(_n5, _t5);

                    // save to out_tm
                    float output_n0[8] = {0.f};
                    _mm256_storeu_ps(output_n0, _n0);
                    float output_n1[8] = {0.f};
                    _mm256_storeu_ps(output_n1, _n1);
                    float output_n2[8] = {0.f};
                    _mm256_storeu_ps(output_n2, _n2);
                    float output_n3[8] = {0.f};
                    _mm256_storeu_ps(output_n3, _n3);
                    float output_n4[8] = {0.f};
                    _mm256_storeu_ps(output_n4, _n4);
                    float output_n5[8] = {0.f};
                    _mm256_storeu_ps(output_n5, _n5);

                    out_tm0[0] = output_n0[0];
                    out_tm0[1] = output_n0[1];
                    out_tm0[2] = output_n0[2];
                    out_tm0[3] = output_n0[3];
                    out_tm1[0] = output_n0[4];
                    out_tm1[1] = output_n0[5];
                    out_tm1[2] = output_n1[0];
                    out_tm1[3] = output_n1[1];
                    out_tm2[0] = output_n1[2];
                    out_tm2[1] = output_n1[3];
                    out_tm2[2] = output_n1[4];
                    out_tm2[3] = output_n1[5];

                    out_tm3[0] = output_n2[0];
                    out_tm3[1] = output_n2[1];
                    out_tm3[2] = output_n2[2];
                    out_tm3[3] = output_n2[3];
                    out_tm4[0] = output_n2[4];
                    out_tm4[1] = output_n2[5];
                    out_tm4[2] = output_n3[0];
                    out_tm4[3] = output_n3[1];
                    out_tm5[0] = output_n3[2];
                    out_tm5[1] = output_n3[3];
                    out_tm5[2] = output_n3[4];
                    out_tm5[3] = output_n3[5];

                    out_tm6[0] = output_n4[0];
                    out_tm6[1] = output_n4[1];
                    out_tm6[2] = output_n4[2];
                    out_tm6[3] = output_n4[3];
                    out_tm7[0] = output_n4[4];
                    out_tm7[1] = output_n4[5];
                    out_tm7[2] = output_n5[0];
                    out_tm7[3] = output_n5[1];
                    out_tm8[0] = output_n5[2];
                    out_tm8[1] = output_n5[3];
                    out_tm8[2] = output_n5[4];
                    out_tm8[3] = output_n5[5];

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                }
            }
        }
    }
    
    // BEGIN dot
    float *top_blob_tm = new float[36 * tiles * outch];
    {
        #pragma omp parallel for num_threads(threads_num)
        for (int r = 0; r < 9; r++)
        {
            int nn_outch = 0;
            int remain_outch_start = 0;

            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = pp * 8;

                float* output0_tm = top_blob_tm + (p + 0) * tiles * 36;
                float* output1_tm = top_blob_tm + (p + 1) * tiles * 36;
                float* output2_tm = top_blob_tm + (p + 2) * tiles * 36;
                float* output3_tm = top_blob_tm + (p + 3) * tiles * 36;
                float* output4_tm = top_blob_tm + (p + 4) * tiles * 36;
                float* output5_tm = top_blob_tm + (p + 5) * tiles * 36;
                float* output6_tm = top_blob_tm + (p + 6) * tiles * 36;
                float* output7_tm = top_blob_tm + (p + 7) * tiles * 36;

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;
                output4_tm = output4_tm + r * 4;
                output5_tm = output5_tm + r * 4;
                output6_tm = output6_tm + r * 4;
                output7_tm = output7_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const float* kptr = kernel_tm2[r] + (p / 8) * inch * 32;
                    const float* r0 = bottom_blob_tm + (tiles * r + i) * inch * 4;

                    float zero_val = 0.f;
                    __m128 _sum0 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum1 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum2 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum3 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum4 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum5 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum6 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum7 = _mm_broadcast_ss(&zero_val);

                    int q = 0;
                    for (; q + 3 < inch; q = q + 4)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _r1 = _mm_loadu_ps(r0 + 4);
                        __m128 _r2 = _mm_loadu_ps(r0 + 8);
                        __m128 _r3 = _mm_loadu_ps(r0 + 12);

                        __m128 _k0 = _mm_loadu_ps(kptr);
                        __m128 _k1 = _mm_loadu_ps(kptr + 4);
                        __m128 _k2 = _mm_loadu_ps(kptr + 8);
                        __m128 _k3 = _mm_loadu_ps(kptr + 12);
                        __m128 _k4 = _mm_loadu_ps(kptr + 16);
                        __m128 _k5 = _mm_loadu_ps(kptr + 20);
                        __m128 _k6 = _mm_loadu_ps(kptr + 24);
                        __m128 _k7 = _mm_loadu_ps(kptr + 28);

                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r0, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r0, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r0, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r0, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r0, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r0, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r0, _k7, _sum7);

                        kptr += 32;
                        _k0 = _mm_loadu_ps(kptr);
                        _k1 = _mm_loadu_ps(kptr + 4);
                        _k2 = _mm_loadu_ps(kptr + 8);
                        _k3 = _mm_loadu_ps(kptr + 12);
                        _k4 = _mm_loadu_ps(kptr + 16);
                        _k5 = _mm_loadu_ps(kptr + 20);
                        _k6 = _mm_loadu_ps(kptr + 24);
                        _k7 = _mm_loadu_ps(kptr + 28);

                        _sum0 = _mm_fmadd_ps(_r1, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r1, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r1, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r1, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r1, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r1, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r1, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r1, _k7, _sum7);

                        kptr += 32;
                        _k0 = _mm_loadu_ps(kptr);
                        _k1 = _mm_loadu_ps(kptr + 4);
                        _k2 = _mm_loadu_ps(kptr + 8);
                        _k3 = _mm_loadu_ps(kptr + 12);
                        _k4 = _mm_loadu_ps(kptr + 16);
                        _k5 = _mm_loadu_ps(kptr + 20);
                        _k6 = _mm_loadu_ps(kptr + 24);
                        _k7 = _mm_loadu_ps(kptr + 28);

                        _sum0 = _mm_fmadd_ps(_r2, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r2, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r2, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r2, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r2, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r2, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r2, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r2, _k7, _sum7);

                        kptr += 32;
                        _k0 = _mm_loadu_ps(kptr);
                        _k1 = _mm_loadu_ps(kptr + 4);
                        _k2 = _mm_loadu_ps(kptr + 8);
                        _k3 = _mm_loadu_ps(kptr + 12);
                        _k4 = _mm_loadu_ps(kptr + 16);
                        _k5 = _mm_loadu_ps(kptr + 20);
                        _k6 = _mm_loadu_ps(kptr + 24);
                        _k7 = _mm_loadu_ps(kptr + 28);

                        _sum0 = _mm_fmadd_ps(_r3, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r3, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r3, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r3, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r3, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r3, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r3, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r3, _k7, _sum7);

                        kptr += 32;
                        r0 += 16;
                    }

                    for (; q < inch; q++)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _k0 = _mm_loadu_ps(kptr);
                        __m128 _k1 = _mm_loadu_ps(kptr + 4);
                        __m128 _k2 = _mm_loadu_ps(kptr + 8);
                        __m128 _k3 = _mm_loadu_ps(kptr + 12);
                        __m128 _k4 = _mm_loadu_ps(kptr + 16);
                        __m128 _k5 = _mm_loadu_ps(kptr + 20);
                        __m128 _k6 = _mm_loadu_ps(kptr + 24);
                        __m128 _k7 = _mm_loadu_ps(kptr + 28);

                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r0, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r0, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r0, _k3, _sum3);
                        _sum4 = _mm_fmadd_ps(_r0, _k4, _sum4);
                        _sum5 = _mm_fmadd_ps(_r0, _k5, _sum5);
                        _sum6 = _mm_fmadd_ps(_r0, _k6, _sum6);
                        _sum7 = _mm_fmadd_ps(_r0, _k7, _sum7);

                        kptr += 32;
                        r0 += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);
                    _mm_storeu_ps(output4_tm, _sum4);
                    _mm_storeu_ps(output5_tm, _sum5);
                    _mm_storeu_ps(output6_tm, _sum6);
                    _mm_storeu_ps(output7_tm, _sum7);

                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                    output4_tm += 36;
                    output5_tm += 36;
                    output6_tm += 36;
                    output7_tm += 36;
                }
            }

            nn_outch = (outch - remain_outch_start) >> 2;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = remain_outch_start + pp * 4;

                float* output0_tm = top_blob_tm + (p + 0) * tiles * inch * 36;
                float* output1_tm = top_blob_tm + (p + 1) * tiles * inch * 36;
                float* output2_tm = top_blob_tm + (p + 2) * tiles * inch * 36;
                float* output3_tm = top_blob_tm + (p + 3) * tiles * inch * 36;

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const float* kptr = kernel_tm2[r] + (p / 8 + (p % 8) / 4) * inch * 32;
                    const float* r0 = bottom_blob_tm + (tiles * r + i) * inch * 4;

                    float zero_val = 0.f;
                    __m128 _sum0 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum1 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum2 = _mm_broadcast_ss(&zero_val);
                    __m128 _sum3 = _mm_broadcast_ss(&zero_val);

                    for (int q = 0; q < inch; q++)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _k0 = _mm_loadu_ps(kptr);
                        __m128 _k1 = _mm_loadu_ps(kptr + 4);
                        __m128 _k2 = _mm_loadu_ps(kptr + 8);
                        __m128 _k3 = _mm_loadu_ps(kptr + 12);

                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);
                        _sum1 = _mm_fmadd_ps(_r0, _k1, _sum1);
                        _sum2 = _mm_fmadd_ps(_r0, _k2, _sum2);
                        _sum3 = _mm_fmadd_ps(_r0, _k3, _sum3);

                        kptr += 16;
                        r0 += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);

                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                }
            }

            remain_outch_start += nn_outch << 2;

            for (int p = remain_outch_start; p < outch; p++)
            {
                float* output0_tm = top_blob_tm + p * tiles * 36;

                output0_tm = output0_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const float* kptr = kernel_tm2[r] + (p / 8 + (p % 8) / 4 + p % 4) * inch * 32;
                    const float* r0 = bottom_blob_tm + (tiles * r + i) * inch * 4;

                    float zero_val = 0.f;
                    __m128 _sum0 = _mm_broadcast_ss(&zero_val);

                    for (int q = 0; q < inch; q++)
                    {
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _k0 = _mm_loadu_ps(kptr);

                        _sum0 = _mm_fmadd_ps(_r0, _k0, _sum0);

                        kptr += 16;
                        r0 += 4;
                    }
                    _mm_storeu_ps(output0_tm, _sum0);

                    output0_tm += 36;
                }
            }
        }
    }
    // END dot

    // BEGIN transform output
    {
        // AT
        // const float itm[4][6] = {
        //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
        // };

        // 0 =	r00 + r01 + r02 + r03 +	r04
        // 1 =	r01 - r02 + 2 * (r03 - r04)
        // 2 =	r01 + r02 + 4 * (r03 + r04)
        // 3 =	r01 - r02 + 8 * (r03 - r04)  + r05

        #pragma omp parallel for num_threads(threads_num)
        for (int p = 0; p < outch; p++)
        {
            float* out_tile = top_blob_tm + p * tiles * 36;
            float* outRow0 = top_blob + p * tiles;
            float* outRow1 = outRow0 + outw;
            float* outRow2 = outRow0 + outw * 2;
            float* outRow3 = outRow0 + outw * 3;

            for (int j = 0; j < colBlocks; j++)
            {
                for (int i = 0; i < rowBlocks; i++)
                {
                    // TODO AVX2
                    float s0[6], s1[6], s2[6], s3[6], s4[6], s5[6];
                    float w0[6], w1[6], w2[6], w3[6];
                    float d0[4], d1[4], d2[4], d3[4], d4[4], d5[4];
                    float o0[4], o1[4], o2[4], o3[4];

                    // load
                    for (int n = 0; n < 6; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n + 6];
                        s2[n] = out_tile[n + 12];
                        s3[n] = out_tile[n + 18];
                        s4[n] = out_tile[n + 24];
                        s5[n] = out_tile[n + 30];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 6; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n] + s3[n] + s4[n];
                        w1[n] = s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n];
                        w2[n] = s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n];
                        w3[n] = s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + s5[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0];
                        d0[1] = w1[0];
                        d0[2] = w2[0];
                        d0[3] = w3[0];
                        d1[0] = w0[1];
                        d1[1] = w1[1];
                        d1[2] = w2[1];
                        d1[3] = w3[1];
                        d2[0] = w0[2];
                        d2[1] = w1[2];
                        d2[2] = w2[2];
                        d2[3] = w3[2];
                        d3[0] = w0[3];
                        d3[1] = w1[3];
                        d3[2] = w2[3];
                        d3[3] = w3[3];
                        d4[0] = w0[4];
                        d4[1] = w1[4];
                        d4[2] = w2[4];
                        d4[3] = w3[4];
                        d5[0] = w0[5];
                        d5[1] = w1[5];
                        d5[2] = w2[5];
                        d5[3] = w3[5];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 4; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n] + d3[n] + d4[n];
                        o1[n] = d1[n] - d2[n] + 2 * d3[n] - 2 * d4[n];
                        o2[n] = d1[n] + d2[n] + 4 * d3[n] + 4 * d4[n];
                        o3[n] = d1[n] - d2[n] + 8 * d3[n] - 8 * d4[n] + d5[n];
                    }
                    // save to top blob tm
                    for (int n = 0; n < 4; n++)
                    {
                        outRow0[n] = o0[n];
                        outRow1[n] = o1[n];
                        outRow2[n] = o2[n];
                        outRow3[n] = o3[n];
                    }

                    out_tile += 36;

                    outRow0 += 4;
                    outRow1 += 4;
                    outRow2 += 4;
                    outRow3 += 4;
                }

                outRow0 += outw * 3;
                outRow1 += outw * 3;
                outRow2 += outw * 3;
                outRow3 += outw * 3;
            }
        }
    }
    // END transform output
    delete[] top_blob_tm;

    delete[] bottom_blob_tm;

    for (int r = 0; r < 9; r++)
    {
        delete[] kernel_tm2[r];
    }

    delete[] kernel_tm;    
}
