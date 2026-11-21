#ifndef MLU_OPERATOR_TYPES_H
#define MLU_OPERATOR_TYPES_H

typedef enum { MLU_OPERATOR_FP16, MLU_OPERATOR_BF16 } mlu_dtype;
typedef enum { MLU_OPERATOR_OK, MLU_OPERATOR_INPUT, MLU_OPERATOR_CNRT, MLU_OPERATOR_CNNL } mlu_error_domain;

typedef struct {
    mlu_error_domain domain;
    int code;
    const char *operation;
} mlu_status;

#endif
