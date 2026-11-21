#include "mlu_rmsnorm.h"

#include <cnnl.h>
#include <math.h>

#include "vendor_status.h"

mlu_status mlu_rmsnorm(int device, mlu_dtype dtype, int rows, int width,
                      float epsilon, const uint16_t *x, const float *gamma,
                      float *output) {
    if (device < 0 || (dtype != MLU_OPERATOR_FP16 && dtype != MLU_OPERATOR_BF16) ||
        rows <= 0 || width <= 0 || !isfinite(epsilon) || epsilon <= 0 ||
        !x || !gamma || !output ||
        (size_t)rows > SIZE_MAX / sizeof(*output) / (size_t)width)
        return (mlu_status){MLU_OPERATOR_INPUT, 0, "validate RMSNorm arguments"};

    const size_t elements = (size_t)rows * (size_t)width;
    const size_t x_bytes = elements * sizeof(*x), y_bytes = elements * sizeof(*output);
    const size_t gamma_bytes = (size_t)width * sizeof(*gamma);
    const int shape[] = {rows, width}, weights[] = {width}, saved[] = {rows};
    const cnnlDataType_t input_type = dtype == MLU_OPERATOR_BF16 ? CNNL_DTYPE_BFLOAT16 : CNNL_DTYPE_HALF;
    mlu_status status = {MLU_OPERATOR_OK, 0, NULL};
    cnrtQueue_t queue = NULL;
    cnnlHandle_t handle = NULL;
    cnnlTensorDescriptor_t xd = NULL, gd = NULL, yd = NULL, rd = NULL;
    cnnlNormDescriptor_t norm = NULL;
    void *dx = NULL, *dg = NULL, *dy = NULL, *dr = NULL, *workspace = NULL;
    size_t workspace_bytes = 0;

    RT(cnrtSetDevice(device));
    RT(cnrtQueueCreate(&queue));
    NL(cnnlCreate(&handle));
    NL(cnnlSetQueue(handle, queue));
    NL(cnnlCreateNormDesc(&norm));
    NL(cnnlCreateTensorDescriptor(&xd));
    NL(cnnlCreateTensorDescriptor(&gd));
    NL(cnnlCreateTensorDescriptor(&yd));
    NL(cnnlCreateTensorDescriptor(&rd));
    NL(cnnlSetTensorDescriptor(xd, CNNL_LAYOUT_ARRAY, input_type, 2, shape));
    NL(cnnlSetTensorDescriptor(gd, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, weights));
    NL(cnnlSetTensorDescriptor(yd, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, shape));
    NL(cnnlSetTensorDescriptor(rd, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, saved));
    NL(cnnlGetRmsNormOpWorkspaceSize(handle, 1, xd, &workspace_bytes));
    RT(cnrtMalloc(&dx, x_bytes));
    RT(cnrtMalloc(&dg, gamma_bytes));
    RT(cnrtMalloc(&dy, y_bytes));
    RT(cnrtMalloc(&dr, (size_t)rows * sizeof(float)));
    if (workspace_bytes)
        RT(cnrtMalloc(&workspace, workspace_bytes));
    RT(cnrtMemcpy(dx, (void *)x, x_bytes, cnrtMemcpyHostToDev));
    RT(cnrtMemcpy(dg, (void *)gamma, gamma_bytes, cnrtMemcpyHostToDev));
    NL(cnnlRmsNormForward_v2(handle, 1, epsilon, norm, xd, dx, gd, dg, NULL,
                            workspace, workspace_bytes, yd, dy, rd, dr));
    RT(cnrtQueueSync(queue));
    RT(cnrtMemcpy(output, dy, y_bytes, cnrtMemcpyDevToHost));

cleanup:
    if (queue) RECORD(MLU_OPERATOR_CNRT, cnrtSuccess, cnrtQueueSync(queue));
    if (workspace) RECORD(MLU_OPERATOR_CNRT, cnrtSuccess, cnrtFree(workspace));
    if (dr) RECORD(MLU_OPERATOR_CNRT, cnrtSuccess, cnrtFree(dr));
    if (dy) RECORD(MLU_OPERATOR_CNRT, cnrtSuccess, cnrtFree(dy));
    if (dg) RECORD(MLU_OPERATOR_CNRT, cnrtSuccess, cnrtFree(dg));
    if (dx) RECORD(MLU_OPERATOR_CNRT, cnrtSuccess, cnrtFree(dx));
    if (rd) RECORD(MLU_OPERATOR_CNNL, CNNL_STATUS_SUCCESS, cnnlDestroyTensorDescriptor(rd));
    if (yd) RECORD(MLU_OPERATOR_CNNL, CNNL_STATUS_SUCCESS, cnnlDestroyTensorDescriptor(yd));
    if (gd) RECORD(MLU_OPERATOR_CNNL, CNNL_STATUS_SUCCESS, cnnlDestroyTensorDescriptor(gd));
    if (xd) RECORD(MLU_OPERATOR_CNNL, CNNL_STATUS_SUCCESS, cnnlDestroyTensorDescriptor(xd));
    if (norm) RECORD(MLU_OPERATOR_CNNL, CNNL_STATUS_SUCCESS, cnnlDestroyNormDesc(norm));
    if (handle) RECORD(MLU_OPERATOR_CNNL, CNNL_STATUS_SUCCESS, cnnlDestroy(handle));
    if (queue) RECORD(MLU_OPERATOR_CNRT, cnrtSuccess, cnrtQueueDestroy(queue));
    return status;
}
