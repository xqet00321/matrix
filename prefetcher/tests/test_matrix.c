#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>

/* provide the implementations of naive_transpose,
 * sse_transpose, sse_prefetch_transpose
 */

#include "../impl/matrix.h"
#include "../impl/stopwatch.h"

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
    &SSEMatrixProvider,
    &SSEPrefetchMatrixProvider,
    &AvxMatrixProvider,
    &AvxPrefetchMatrixProvider
};

// Default array size is 4096, ARRAY_SIZE_I is how many times of
// 64 size of width should add to array size
//
// array size = 4096 + ARRAY_SIZE_I * 64;
#ifndef ARRAY_SIZE_I
#define ARRAY_SIZE_I  0
#endif

int main()
{
    int TEST_W = 4096 + ARRAY_SIZE_I * 64;
    int TEST_H = 4096 + ARRAY_SIZE_I * 64;

    /* Matrix algorithm declaration */
#ifdef avx_prefetch
    MatrixAlgo *algo = matrix_providers[4];
#elif avx
    MatrixAlgo *algo = matrix_providers[3];
#elif sse_prefetch
    MatrixAlgo *algo = matrix_providers[2];
#elif sse
    MatrixAlgo *algo = matrix_providers[1];
#else
    MatrixAlgo *algo = matrix_providers[0];
#endif

    /* verify the result of 4x4 matrix */
    {
        float testin[4][4] = {
            {  0,  1,  2,  3,},
            {  4,  5,  6,  7,},
            {  8,  9, 10, 11,},
            { 12, 13, 14, 15,},
        };

        float expected[4][4] = {
            { 0, 4,  8, 12,},
            { 1, 5,  9, 13,},
            { 2, 6, 10, 14,},
            { 3, 7, 11, 15,},
        };

        int testin_size = sizeof(testin) / sizeof(testin[0][0]);
        int expected_size = sizeof(expected) / sizeof(expected[0][0]);

        Matrix *matrix, *ans, *expect;

        /* expected answer */
        expect = algo->create(4, 4);
        algo->assign(expect, *expected, expected_size);

        /* create & assign matrix */
        ans = algo->create(4, 4);
        matrix = algo->create(4, 4);
        algo->assign(matrix, *testin, testin_size);

        /* transpose */
        algo->transpose(ans,matrix);

        /* correctness check */
        assert(1 == algo->equal(ans, expect) &&
               "Verification fails");
    }

    /* verify the result of 8x8 matrix */
    {
        float testin[8][8] = {
            {  0,  1,  2,  3,  4,  5,  6,  7,},
            {  8,  9, 10, 11, 12, 13, 14, 15,},
            { 16, 17, 18, 19, 20, 21, 22, 23,},
            { 24, 25, 26, 27, 28, 29, 30, 31,},
            { 32, 33, 34, 35, 36, 37, 38, 39,},
            { 40, 41, 42, 43, 44, 45, 46, 47,},
            { 48, 49, 50, 51, 52, 53, 54, 55,},
            { 56, 57, 58, 59, 60, 61, 62, 63,},
        };

        float expected[8][8] = {
            { 0,  8, 16, 24, 32, 40, 48, 56,},
            { 1,  9, 17, 25, 33, 41, 49, 57,},
            { 2, 10, 18, 26, 34, 42, 50, 58,},
            { 3, 11, 19, 27, 35, 43, 51, 59,},
            { 4, 12, 20, 28, 36, 44, 52, 60,},
            { 5, 13, 21, 29, 37, 45, 53, 61,},
            { 6, 14, 22, 30, 38, 46, 54, 62,},
            { 7, 15, 23, 31, 39, 47, 55, 63,},
        };

        int testin_size = sizeof(testin) / sizeof(testin[0][0]);
        int expected_size = sizeof(expected) / sizeof(expected[0][0]);

        Matrix *matrix, *ans, *expect;

        /* expected answer */
        expect = algo->create(8, 8);
        algo->assign(expect, *expected, expected_size);

        /* create & assign matrix */
        ans = algo->create(8, 8);
        matrix = algo->create(8, 8);
        algo->assign(matrix, *testin, testin_size);

        /* transpose */
        algo->transpose(ans,matrix);

        /* correctness check */
        assert(1 == algo->equal(ans, expect) &&
               "Verification fails");
    }

    /* Performance test */
    {
        float *src  = (float *) malloc(sizeof(float) * TEST_W * TEST_H);

        srand(time(NULL));
        for (int i = 0; i < TEST_H; ++i) {
            for (int j = 0; j < TEST_W; ++j) {
                src[i * TEST_W + j] = rand();
            }
        }

        int src_size = TEST_W * TEST_H;

        Matrix *ans, *matrix;

        /* create & assign matrix */
        ans = algo->create(TEST_W, TEST_H);
        matrix = algo->create(TEST_W, TEST_H);
        algo->assign(matrix, src, src_size);

        /* setup timer */
        watch_p ctx = Stopwatch.create();
        assert(ctx && "Clock setup fail!");

        /* transpose & measure time */
        Stopwatch.start(ctx);
        algo->transpose(ans, matrix);
        double now = Stopwatch.read(ctx);

        /* print result */
        printf("%g\n", now);
        Stopwatch.destroy(ctx);
    }

    return 0;
}
