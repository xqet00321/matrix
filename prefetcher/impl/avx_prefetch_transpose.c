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

static void avx_prefetch_transpose(Matrix *dst, const Matrix *src)
{
    assert(dst->col == src->col && dst->row == src->row
           && "Matrix size is different!");

    int w = dst->col;
    int h = dst->row;

    for (int x = 0; x < w; x += 8) {
        for (int y = 0; y < h; y += 8) {
#define AVXPFDIST  16
            _mm_prefetch(PRIV(src)+(y + AVXPFDIST + 0) *w + x, _MM_HINT_T1);
            _mm_prefetch(PRIV(src)+(y + AVXPFDIST + 1) *w + x, _MM_HINT_T1);
            _mm_prefetch(PRIV(src)+(y + AVXPFDIST + 2) *w + x, _MM_HINT_T1);
            _mm_prefetch(PRIV(src)+(y + AVXPFDIST + 3) *w + x, _MM_HINT_T1);
            _mm_prefetch(PRIV(src)+(y + AVXPFDIST + 4) *w + x, _MM_HINT_T1);
            _mm_prefetch(PRIV(src)+(y + AVXPFDIST + 5) *w + x, _MM_HINT_T1);
            _mm_prefetch(PRIV(src)+(y + AVXPFDIST + 6) *w + x, _MM_HINT_T1);
            _mm_prefetch(PRIV(src)+(y + AVXPFDIST + 7) *w + x, _MM_HINT_T1);

            __m256i I0 = _mm256_loadu_si256((__m256i *)(PRIV(src) + (y + 0) * w + x));
            __m256i I1 = _mm256_loadu_si256((__m256i *)(PRIV(src) + (y + 1) * w + x));
            __m256i I2 = _mm256_loadu_si256((__m256i *)(PRIV(src) + (y + 2) * w + x));
            __m256i I3 = _mm256_loadu_si256((__m256i *)(PRIV(src) + (y + 3) * w + x));
            __m256i I4 = _mm256_loadu_si256((__m256i *)(PRIV(src) + (y + 4) * w + x));
            __m256i I5 = _mm256_loadu_si256((__m256i *)(PRIV(src) + (y + 5) * w + x));
            __m256i I6 = _mm256_loadu_si256((__m256i *)(PRIV(src) + (y + 6) * w + x));
            __m256i I7 = _mm256_loadu_si256((__m256i *)(PRIV(src) + (y + 7) * w + x));

            __m256i T0 = _mm256_unpacklo_epi32(I0, I1);
            __m256i T1 = _mm256_unpacklo_epi32(I2, I3);
            __m256i T2 = _mm256_unpackhi_epi32(I0, I1);
            __m256i T3 = _mm256_unpackhi_epi32(I2, I3);
            __m256i T4 = _mm256_unpacklo_epi32(I4, I5);
            __m256i T5 = _mm256_unpacklo_epi32(I6, I7);
            __m256i T6 = _mm256_unpackhi_epi32(I4, I5);
            __m256i T7 = _mm256_unpackhi_epi32(I6, I7);

            I0 = _mm256_unpacklo_epi64(T0, T1);
            I1 = _mm256_unpackhi_epi64(T0, T1);
            I2 = _mm256_unpacklo_epi64(T2, T3);
            I3 = _mm256_unpackhi_epi64(T2, T3);
            I4 = _mm256_unpacklo_epi64(T4, T5);
            I5 = _mm256_unpackhi_epi64(T4, T5);
            I6 = _mm256_unpacklo_epi64(T6, T7);
            I7 = _mm256_unpackhi_epi64(T6, T7);

            T0 = _mm256_permute2x128_si256(I0, I4, 0x20);
            T1 = _mm256_permute2x128_si256(I1, I5, 0x20);
            T2 = _mm256_permute2x128_si256(I2, I6, 0x20);
            T3 = _mm256_permute2x128_si256(I3, I7, 0x20);
            T4 = _mm256_permute2x128_si256(I0, I4, 0x31);
            T5 = _mm256_permute2x128_si256(I1, I5, 0x31);
            T6 = _mm256_permute2x128_si256(I2, I6, 0x31);
            T7 = _mm256_permute2x128_si256(I3, I7, 0x31);

            _mm256_storeu_si256((__m256i *)(PRIV(dst) + ((x + 0) * h) + y), T0);
            _mm256_storeu_si256((__m256i *)(PRIV(dst) + ((x + 1) * h) + y), T1);
            _mm256_storeu_si256((__m256i *)(PRIV(dst) + ((x + 2) * h) + y), T2);
            _mm256_storeu_si256((__m256i *)(PRIV(dst) + ((x + 3) * h) + y), T3);
            _mm256_storeu_si256((__m256i *)(PRIV(dst) + ((x + 4) * h) + y), T4);
            _mm256_storeu_si256((__m256i *)(PRIV(dst) + ((x + 5) * h) + y), T5);
            _mm256_storeu_si256((__m256i *)(PRIV(dst) + ((x + 6) * h) + y), T6);
            _mm256_storeu_si256((__m256i *)(PRIV(dst) + ((x + 7) * h) + y), T7);
        }
    }

}

static void println(Matrix *thiz)
{
    int col = thiz->col;
    for (int i = 0; i < thiz->row; i++) {
        for (int j = 0; j < thiz->col; j++) {
            printf("%2g ", PRIV(thiz)[i*col + j]);
        }
        printf("\n");
    }
    printf("\n");
}

MatrixAlgo AvxPrefetchMatrixProvider = {
    .create = create,
    .assign = assign,
    .equal = equal,
    .transpose = avx_prefetch_transpose,
    .println = println,
};
