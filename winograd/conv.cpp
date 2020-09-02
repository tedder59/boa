#include "conv.h"
#include <assert.h>
#include <omp.h>

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
                    for (int a = 0; a < spatial_size; a++)
                    {
                        m_buffer[a] = ptr_weight[a] * BTdTB[a];
                    }

                    for (int ch = 1; ch < in_channels; ch++)
                    {
                        const float* ptr0 = ptr_weight + ch * spatial_size;
                        const float* ptr1 = BTdTB + ch * spatial_size;

                        for (int a = 0; a < spatial_size; a++)
                        {
                            m_buffer[a] += ptr0[a] * ptr1[a];
                        }
                    }

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

        // roi transpose
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

            float val = 0.f;
            for (int i = 0; i < s; i++)
            {
                val += ptr_a[i] * ptr_b[i];
            }

            ptr[x] = val;
        }   
    }  
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

        at = {1.f, 0.f, 0.f, 0.f,
              1.f, 1.f, 1.f, 1.f,
              1.f, -1.f, 1.f, -1.f,
              1.f, 2.f, 4.f, 8.f,
              1.f, -2.f, 4.f, -8.f,
              0.f, 0.f, 0.f, 1.f};

        bt = {4.f, 0.f, 0.f, 0.f, 0.f, 0.f,
              0.f, -4.f, 4.f, -2.f, 2.f, 4.f,
              -5.f, -4.f, -4.f, -1.f, -1.f, 0.f,
              0.f, 1.f, -1.f, 2.f, -2.f, -5.f,
              1.f, 1.f, 1.f, 1.f, 1.f, 0.f,
              0.f, 0.f, 0.f, 0.f, 0.f, 1.f};
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

        at = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f,
              1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
              1.f, -1.f, 1.f, -1.f, 1.f, -1.f,
              1.f, 2.f, 4.f, 8.f, 16.f, 32.f,
              1.f, -2.f, 4.f, -8.f, 16.f, -32.f,
              1.f, 3.f, 9.f, 27.f, 81.f, 243.f,
              1.f, -3.f, 9.f, -27.f, 81.f, -243.f,
              0.f, 0.f, 0.f, 0.f, 0.f, 1.f};

        bt = {36.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
              0.f, 36.f, -36.f, 18.f, -18.f, 12.f, -12.f, -36.f,
              -49.f, 36.f, 36.f, 9.f, 9.f, 4.f, 4.f, 0.f,
              0.f, -13.f, 13.f, -20.f, 20.f, -15.f, 15.f, 49.f,
              14.f, -13.f, -13.f, -10.f, -10.f, -5.f, -5.f, 0.f,
              0.f, 1.f, -1.f, 2.f, -2.f, 3.f, -3.f, -14.f,
              -1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f,
              0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f};

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