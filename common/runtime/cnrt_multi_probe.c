#include <cnrt.h>
#include <stdio.h>
#include <string.h>

enum { CARDS = 8, BYTES = 4096 };
static int check(cnrtRet_t rc, int card, const char *op) {
    printf("ordinal=%d %s status=%u\n", card, op, (unsigned)rc);
    if (rc != cnrtSuccess)
        fprintf(stderr, "%s: %s\n", op, cnrtGetErrorStr(rc));
    return rc == cnrtSuccess;
}
#define CHECK(call) check((call), i, #call)

int main(void) {
    void *buffers[CARDS] = {0};
    cnrtQueue_t queues[CARDS] = {0};
    unsigned char pattern[CARDS][BYTES], result[BYTES];
    unsigned count = 0;
    if (cnrtGetDeviceCount(&count) != cnrtSuccess || count != CARDS)
        return 1;
    cnrtRet_t invalid = cnrtSetDevice(CARDS);
    printf("invalid_ordinal=%d status=%u\n", CARDS, (unsigned)invalid);
    if (invalid != cnrtErrorArgsInvalid && invalid != cnrtErrorDeviceInvalid) return 1;
    int passed = 0;
    for (int i = 0; i < CARDS; ++i) {
        for (unsigned j = 0; j < BYTES; ++j)
            pattern[i][j] = (unsigned char)(31*j + 17*i);
        if (!CHECK(cnrtSetDevice(i)) || !CHECK(cnrtQueueCreate(&queues[i])) ||
            !CHECK(cnrtMalloc(&buffers[i], BYTES)) ||
            !CHECK(cnrtMemcpy(buffers[i], pattern[i], BYTES, cnrtMemcpyHostToDev)))
            goto cleanup;
    }
    puts("eight buffers and queues live simultaneously");
    for (int i = CARDS-1; i >= 0; --i) {
        if (!CHECK(cnrtSetDevice(i)) || !CHECK(cnrtQueueSync(queues[i])) ||
            !CHECK(cnrtMemcpy(result, buffers[i], BYTES, cnrtMemcpyDevToHost)) ||
            memcmp(result, pattern[i], BYTES)) {
            fprintf(stderr, "readback failed for ordinal %d\n", i);
            goto cleanup;
        }
    }
    passed = 1;
cleanup:
    for (int i = 0; i < CARDS; ++i) {
        if (!(buffers[i] || queues[i])) continue;
        if (!CHECK(cnrtSetDevice(i))) { passed = 0; continue; }
        if (buffers[i]) passed &= CHECK(cnrtFree(buffers[i]));
        if (queues[i]) passed &= CHECK(cnrtQueueDestroy(queues[i]));
    }
    if (passed) puts("PASS: eight simultaneous allocations, queues and unique patterns");
    return !passed;
}
