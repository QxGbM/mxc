
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void dtrsmlt_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb);

void dtrsmr_right(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb);

void dtrsml_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb);

void dtrsmlt_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb);

void dtrsmr_left(int64_t m, int64_t n, const double* a, int64_t lda, double* b, int64_t ldb);

void Cdgemv(char ta, int64_t m, int64_t n, double alpha, const double* a, int64_t lda, const double* x, int64_t incx, double beta, double* y, int64_t incy);

void Cdgemm(char ta, char tb, int64_t m, int64_t n, int64_t k, double alpha, const double* a, int64_t lda, const double* b, int64_t ldb, double beta, double* c, int64_t ldc);

void Cdcopy(int64_t n, const double* x, int64_t incx, double* y, int64_t incy);

void Cdscal(int64_t n, double alpha, double* x, int64_t incx);

void Cdaxpy(int64_t n, double alpha, const double* x, int64_t incx, double* y, int64_t incy);

void Cddot(int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result);

void Cidamax(int64_t n, const double* x, int64_t incx, int64_t* ida);

void Cdnrm2(int64_t n, const double* x, int64_t incx, double* nrm_out);

int64_t* getFLOPS();

#ifdef __cplusplus
}
#endif
