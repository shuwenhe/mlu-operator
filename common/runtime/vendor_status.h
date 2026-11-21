#ifndef MLU_OPERATOR_VENDOR_STATUS_H
#define MLU_OPERATOR_VENDOR_STATUS_H

/* Retain the first failure, including the original vendor status. */
#define RECORD(domain_value, success, call) do { \
    const int code = (int)(call); \
    if (code != (int)(success) && status.domain == MLU_OPERATOR_OK) \
        status = (mlu_status){domain_value, code, #call}; \
} while (0)
#define RT(call) do { \
    RECORD(MLU_OPERATOR_CNRT, cnrtSuccess, call); \
    if (status.domain != MLU_OPERATOR_OK) goto cleanup; \
} while (0)
#define NL(call) do { \
    RECORD(MLU_OPERATOR_CNNL, CNNL_STATUS_SUCCESS, call); \
    if (status.domain != MLU_OPERATOR_OK) goto cleanup; \
} while (0)

#endif
