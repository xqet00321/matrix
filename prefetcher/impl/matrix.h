#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdbool.h>

/* predefined shortcut */
#define DECLARE_MATRIX(row, col) \
    typedef struct { float values[col][row]; } Mat ## row ## x ## col
DECLARE_MATRIX(3, 3);
DECLARE_MATRIX(4, 4);

typedef struct {
    int row, col;
    void *priv;
} Matrix;

typedef struct {
    Matrix* (*create)(int row, int col);
    void (*assign)(Matrix *thiz, float* data, int data_size);
    bool (*equal)(const Matrix *l, const Matrix *r);
    void (*transpose)(Matrix *dst, const Matrix *src);
    void (*println)(Matrix *thiz);
} MatrixAlgo;

/* Available matrix providers */
extern MatrixAlgo NaiveMatrixProvider;
extern MatrixAlgo SSEMatrixProvider;
extern MatrixAlgo SSEPrefetchMatrixProvider;
extern MatrixAlgo AvxMatrixProvider;
extern MatrixAlgo AvxSetMatrixProvider;
extern MatrixAlgo AvxPrefetchMatrixProvider;
#endif /* MATRIX_H_ */
