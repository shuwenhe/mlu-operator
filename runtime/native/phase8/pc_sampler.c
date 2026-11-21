// CPU-only audit. The handler records a PC and avoids symbol resolution or I/O.
#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <ucontext.h>

enum { capacity = 65536 };
static uintptr_t pcs[capacity];
static unsigned samples;

static void record(int signal, siginfo_t *info, void *context) {
    (void)signal;
    (void)info;
    ucontext_t *state = context;
    unsigned index = __atomic_fetch_add(&samples, 1, __ATOMIC_RELAXED);
    if (index < capacity) pcs[index] = state->uc_mcontext.gregs[REG_RIP];
}

int begin_samples(void) {
    __atomic_store_n(&samples, 0, __ATOMIC_RELAXED);
    struct sigaction action = {0};
    action.sa_sigaction = record;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    if (sigaction(SIGPROF, &action, 0)) return -1;
    struct itimerval timer = {{0, 401}, {0, 401}};
    return setitimer(ITIMER_PROF, &timer, 0);
}

unsigned end_samples(const char *name) {
    struct itimerval disabled = {0};
    if (setitimer(ITIMER_PROF, &disabled, 0)) return 0;
    FILE *output = fopen(name, "w");
    if (!output) return 0;
    unsigned count = __atomic_load_n(&samples, __ATOMIC_RELAXED);
    if (count > capacity) count = capacity;
    for (unsigned i = 0; i < count; ++i) fprintf(output, "%lx\n", (unsigned long)pcs[i]);
    return fclose(output) ? 0 : count;
}
