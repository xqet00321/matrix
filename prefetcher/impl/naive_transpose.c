#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

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

static void naive_transpose(Matrix *dst, const Matrix *src)
{
    assert(dst->col == src->col && dst->row == src->row
           && "Matrix size is different!");

    for (int i = 0; i < dst->row; i++)
        for (int j = 0; j < dst->col; j++)
            PRIV(dst)[i*dst->col + j] = PRIV(src)[j*dst->col + i];
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

MatrixAlgo NaiveMatrixProvider = {
    .create = create,
    .assign = assign,
    .equal = equal,
    .transpose = naive_transpose,
    .println = println,
};