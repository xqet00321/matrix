#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <immintrin.h>

#define PRIV(x) \
    ((float *) ((x)->priv))

static float* matrix_alloc(int row, int col)
{
    float *values = (float *)calloc(row * col, sizeof(float));
    return values;
}

static Matrix* create(int row, int col)
{
    Matrix *matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->row = row;
    matrix->col = col;
    matrix->priv = matrix_alloc(row, col);
    return matrix;
}

static void assign(Matrix *thiz, float* data, int data_size)
{
    assert(thiz->row * thiz->col == data_size &&
           "Data size different from matrix size");

    for (int i = 0; i < thiz->row; i++)
        for (int j = 0; j < thiz->col; j++)
            PRIV(thiz)[i*thiz->col + j] = data[i*thiz->col + j];
}

static bool equal(const Matrix *l, const Matrix *r)
{
    const float epsilon = 1 / 10000.0;

    assert(l->row == r->row && l->col == r->col && "Matrix size is different");
    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < l->col; j++)
            if (PRIV(l)[i * l->col + j] + epsilon < PRIV(r)[i * r->col + j] ||
                    PRIV(r)[i * r->col + j] + epsilon < PRIV(l)[i * l->col + j])
                return false;
    return true;
}

void sse_prefetch_transpose(Matrix *dst, const Matrix *src)
{
    assert(dst->col == src->col && dst->row == src->row
           && "Matrix size is different!");

    int w = dst->col;
    int h = dst->row;

    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {
#define PFDIST  8
            _mm_prefetch(src->priv +(y + PFDIST + 0) *w * 4 + x*4, _MM_HINT_T1);
            _mm_prefetch(src->priv +(y + PFDIST + 1) *w * 4 + x*4, _MM_HINT_T1);
            _mm_prefetch(src->priv +(y + PFDIST + 2) *w * 4 + x*4, _MM_HINT_T1);
            _mm_prefetch(src->priv +(y + PFDIST + 3) *w * 4 + x*4, _MM_HINT_T1);

            __m128i I0 = _mm_loadu_si128 ((__m128i *)(src->priv + (y + 0) * w * 4 + x*4));
            __m128i I1 = _mm_loadu_si128 ((__m128i *)(src->priv + (y + 1) * w * 4 + x*4));
            __m128i I2 = _mm_loadu_si128 ((__m128i *)(src->priv + (y + 2) * w * 4 + x*4));
            __m128i I3 = _mm_loadu_si128 ((__m128i *)(src->priv + (y + 3) * w * 4 + x*4));
            __m128i T0 = _mm_unpacklo_epi32(I0, I1);
            __m128i T1 = _mm_unpacklo_epi32(I2, I3);
            __m128i T2 = _mm_unpackhi_epi32(I0, I1);
            __m128i T3 = _mm_unpackhi_epi32(I2, I3);
            I0 = _mm_unpacklo_epi64(T0, T1);
            I1 = _mm_unpackhi_epi64(T0, T1);
            I2 = _mm_unpacklo_epi64(T2, T3);
            I3 = _mm_unpackhi_epi64(T2, T3);
            _mm_storeu_si128((__m128i *)(dst->priv + ((x + 0) * h * 4) + y*4), I0);
            _mm_storeu_si128((__m128i *)(dst->priv + ((x + 1) * h * 4) + y*4), I1);
            _mm_storeu_si128((__m128i *)(dst->priv + ((x + 2) * h * 4) + y*4), I2);
            _mm_storeu_si128((__m128i *)(dst->priv + ((x + 3) * h * 4) + y*4), I3);
        }
    }
}

static void println(Matrix *thiz)
{
    int col = thiz->col;
    for (int i = 0; i < thiz->row; i++) {
        for (int j = 0; j < thiz->col; j++) {
            printf("%5g ", PRIV(thiz)[i*col + j]);
        }
        printf("\n");
    }
    printf("\n");
}

MatrixAlgo SSEPrefetchMatrixProvider = {
    .create = create,
    .assign = assign,
    .equal = equal,
    .transpose = sse_prefetch_transpose,
    .println = println,
};