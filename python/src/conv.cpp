#include "winograd/conv.h"
#include <torch/extension.h>
#include <ATen/TensorUtils.h>

at::Tensor conv_direct(
    const at::Tensor& input,
    const at::Tensor& weight
)
{
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);

    auto num_output = weight.size(0);
    auto kernel_size = weight.size(2);

    auto output_h = input_h - kernel_size + 1;
    auto output_w = input_w - kernel_size + 1;

    at::Tensor output = at::zeros(
        {batch_size, num_output, output_h, output_w},
        input.options()
    );

    DirectConv conv;
    conv(batch_size, num_output, in_channels,
         input.contiguous().data_ptr<float>(),
         input_h, input_w,
         weight.contiguous().data_ptr<float>(),
         kernel_size,
         output.data_ptr<float>());

    return output;
}

at::Tensor conv_gemm(
    const at::Tensor& input,
    const at::Tensor& weight
)
{
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);

    auto num_output = weight.size(0);
    auto kernel_size = weight.size(2);

    auto output_h = input_h - kernel_size + 1;
    auto output_w = input_w - kernel_size + 1;

    at::Tensor output = at::zeros(
        {batch_size, num_output, output_h, output_w},
        input.options()
    );

    GemmConv conv;
    conv(batch_size, num_output, in_channels,
         input.contiguous().data_ptr<float>(),
         input_h, input_w,
         weight.contiguous().data_ptr<float>(),
         kernel_size,
         output.data_ptr<float>());

    return output;
}

at::Tensor conv_winograd(
    int n, int r,
    const at::Tensor& input,
    const at::Tensor& weight
)
{
    WinogradConv conv(n, r);

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);

    auto num_output = weight.size(0);
    auto kernel_size = weight.size(2);

    auto output_h = input_h - kernel_size + 1;
    auto output_w = input_w - kernel_size + 1;

    at::Tensor output = at::zeros(
        {batch_size, num_output, output_h, output_w},
        input.options()
    );

    conv(batch_size, num_output, in_channels,
         input.contiguous().data_ptr<float>(),
         input_h, input_w,
         weight.contiguous().data_ptr<float>(),
         kernel_size,
         output.data_ptr<float>());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_direct", &conv_direct, "conv_direct");
    m.def("conv_gemm", &conv_gemm, "conv_gemm");
    m.def("conv_winograd", &conv_winograd, "conv_winograd");
}