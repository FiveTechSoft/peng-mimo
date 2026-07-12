#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

/* Child runs st_init on dir; parent expects non-zero exit (malformed). */
static int expect_st_init_fail(const char *dir) {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return -1; }
    if (pid == 0) {
        shards S;
        /* st_init exits(1) on bad files */
        st_init(&S, dir);
        _exit(0);   /* should not reach: valid empty dir has no tensors but succeeds */
    }
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) { perror("waitpid"); return -1; }
    if (WIFEXITED(st) && WEXITSTATUS(st) != 0) return 1; /* failed as expected */
    return 0;
}

static int write_file(const char *path, const void *buf, size_t n) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;
    ssize_t w = write(fd, buf, n);
    close(fd);
    return (w == (ssize_t)n) ? 0 : -1;
}

int main(void) {
    CHECK(bf16_to_f32(0x3f80) == 1.0f);
    CHECK(bf16_to_f32(0xc020) == -2.5f);
    CHECK(f16_to_f32(0x3c00) == 1.0f);
    CHECK(f16_to_f32(0xc100) == -2.5f);
    CHECK(f16_to_f32(0x0001) > 0.0f);
    CHECK(isinf(f16_to_f32(0x7c00)));
    CHECK(st_hash("tensor.weight") == st_hash("tensor.weight"));
    CHECK(st_hash("tensor.weight") != st_hash("tensor.bias"));
    CHECK(st_dtype_esz(0) == 2);
    CHECK(st_dtype_esz(1) == 2);
    CHECK(st_dtype_esz(2) == 4);
    CHECK(st_dtype_esz(3) == 1);

    /* F-01: oversized header length must not be trusted */
    {
        char dir[] = "/tmp/peng_st_badXXXXXX";
        CHECK(mkdtemp(dir) != NULL);
        char path[512];
        snprintf(path, sizeof(path), "%s/bad.safetensors", dir);
        uint8_t blob[16];
        memset(blob, 0, sizeof(blob));
        /* hlen = 0x100000000ULL claimed via only 8 bytes... use huge hlen */
        uint64_t hlen = (uint64_t)1 << 40;   /* 1 TiB header claim */
        memcpy(blob, &hlen, 8);
        CHECK(write_file(path, blob, 16) == 0);
        CHECK(expect_st_init_fail(dir) == 1);
        unlink(path);
        rmdir(dir);
    }

    /* F-01: header longer than file */
    {
        char dir[] = "/tmp/peng_st_shortXXXXXX";
        CHECK(mkdtemp(dir) != NULL);
        char path[512];
        snprintf(path, sizeof(path), "%s/short.safetensors", dir);
        uint8_t blob[12];
        uint64_t hlen = 1000;   /* claims 1000-byte JSON, file only 12 bytes */
        memcpy(blob, &hlen, 8);
        blob[8] = '{'; blob[9] = '}'; blob[10] = 0; blob[11] = 0;
        CHECK(write_file(path, blob, 12) == 0);
        CHECK(expect_st_init_fail(dir) == 1);
        unlink(path);
        rmdir(dir);
    }

    puts("safetensors primitive tests: ok");
    return 0;
}
