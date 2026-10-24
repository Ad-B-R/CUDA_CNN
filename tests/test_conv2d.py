import itertools

import pytest
import torch
import torch.nn.functional as F

import cuda_cnn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

DEV = "cuda"
ATOL = 1e-4
RTOL = 1e-4

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

# (stride, padding, dilation, kernel) combos, including asymmetric ones
CONFIGS = list(itertools.product(
    [(1, 1), (2, 2), (2, 3)],
    [(0, 0), (1, 1), (2, 0)],
    [(1, 1), (2, 1)],
    [(3, 3), (1, 1), (2, 3), (5, 5)],
))


def reference(x, W, b, stride, padding, dilation):
    x = x.detach().clone().requires_grad_()
    W = W.detach().clone().requires_grad_()
    b = b.detach().clone().requires_grad_() if b is not None else None
    y = F.conv2d(x, W, b, stride=stride, padding=padding, dilation=dilation)
    return x, W, b, y


@pytest.mark.parametrize("stride,padding,dilation,kernel", CONFIGS)
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_matches_torch(stride, padding, dilation, kernel, bias):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 11, 9, device=DEV)
    W = torch.randn(5, 4, *kernel, device=DEV)
    b = torch.randn(5, device=DEV) if bias else None
    if 11 + 2 * padding[0] < dilation[0] * (kernel[0] - 1) + 1 or 9 + 2 * padding[1] < dilation[1] * (kernel[1] - 1) + 1:
        pytest.skip("kernel larger than padded input")

    xr, Wr, br, ref = reference(x, W, b, stride, padding, dilation)
    y = cuda_cnn.conv2d_fwd(x, W, b, stride, padding, dilation)
    torch.testing.assert_close(y, ref, atol=ATOL, rtol=RTOL)

    dy = torch.randn_like(ref)
    ref.backward(dy)
    dx, dW, db = cuda_cnn.conv2d_bwd(dy, x, W, stride, padding, dilation)
    torch.testing.assert_close(dx, xr.grad, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(dW, Wr.grad, atol=ATOL, rtol=RTOL)
    if bias:
        torch.testing.assert_close(db, br.grad, atol=ATOL, rtol=RTOL)


def test_conv2d_cnn_sized_layer():
    # a realistic layer: batch 32, 3 -> 16 channels, 32x32 images, 3x3 "same" conv
    torch.manual_seed(0)
    x = torch.randn(32, 3, 32, 32, device=DEV)
    W = torch.randn(16, 3, 3, 3, device=DEV) * 0.1
    b = torch.randn(16, device=DEV)

    xr, Wr, br, ref = reference(x, W, b, (1, 1), (1, 1), (1, 1))
    y = cuda_cnn.conv2d_fwd(x, W, b, padding=[1, 1])
    torch.testing.assert_close(y, ref, atol=1e-3, rtol=1e-3)

    dy = torch.randn_like(ref)
    ref.backward(dy)
    dx, dW, db = cuda_cnn.conv2d_bwd(dy, x, W, padding=[1, 1])
    torch.testing.assert_close(dx, xr.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(dW, Wr.grad, atol=1e-2, rtol=1e-3)  # sums over 32*32*32 terms
    torch.testing.assert_close(db, br.grad, atol=1e-2, rtol=1e-3)


def test_conv2d_non_contiguous_input():
    x = torch.randn(2, 9, 9, 3, device=DEV).permute(0, 3, 1, 2)  # NHWC storage viewed as NCHW
    W = torch.randn(4, 3, 3, 3, device=DEV)
    assert not x.is_contiguous()
    torch.testing.assert_close(cuda_cnn.conv2d_fwd(x, W), F.conv2d(x, W), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("call,match", [
    (lambda x, W: cuda_cnn.conv2d_fwd(x.cpu(), W), "CUDA"),
    (lambda x, W: cuda_cnn.conv2d_fwd(x.double(), W), "float32"),
    (lambda x, W: cuda_cnn.conv2d_fwd(x[:, :2], W), "channels"),
    (lambda x, W: cuda_cnn.conv2d_fwd(x, W, stride=[0, 1]), "stride"),
    (lambda x, W: cuda_cnn.conv2d_fwd(x, W, padding=[-1, 0]), "padding"),
    (lambda x, W: cuda_cnn.conv2d_fwd(x, W, stride=[1]), "pair"),
    (lambda x, W: cuda_cnn.conv2d_fwd(x, torch.randn(4, 3, 9, 9, device=DEV)), "larger"),
    (lambda x, W: cuda_cnn.conv2d_fwd(x, W, torch.randn(5, device=DEV)), "b of shape"),
    (lambda x, W: cuda_cnn.conv2d_bwd(torch.randn(2, 4, 5, 5, device=DEV), x, W), "dy of shape"),
])
def test_conv2d_rejects_bad_inputs(call, match):
    x = torch.randn(2, 3, 6, 6, device=DEV)
    W = torch.randn(4, 3, 3, 3, device=DEV)
    with pytest.raises(RuntimeError, match=match):
        call(x, W)
