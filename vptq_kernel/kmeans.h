#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cblas.h>
#include <cstring>
#include <memory>

#include <pybind11/numpy.h>

struct matview_2d {
    float* data;
    int rows;
    int cols;

    matview_2d(float* data, int rows, int cols)
        : data(data)
        , rows(rows)
        , cols(cols)
    {
    }

    float* operator[](int i) const
    {
        return data + i * cols;
    }

    float& operator()(int i, int j) const
    {
        return data[i * cols + j];
    }

    static matview_2d from_array_f32(pybind11::array_t<float, pybind11::array::c_style>& src)
    {
        assert(src.ndim() == 2);
        return matview_2d(src.mutable_data(), src.shape(0), src.shape(1));
    }

    static matview_2d from_unchecked_array_f32_2d(pybind11::detail::unchecked_mutable_reference<float, 2>& src)
    {
        assert(src.ndim() == 2);
        return matview_2d(&src(0, 0), src.shape(0), src.shape(1));
    }

    void inplace_zero()
    {
        std::fill(data, data + rows * cols, 0.0f);
    }

    void inplace_copy_subview(const matview_2d& src, int i1, int i2, int j1, int j2)
    {
        assert(i1 >= 0 && i1 < src.rows && i2 >= 0 && i2 <= src.rows && j1 >= 0 && j1 < src.cols && j2 >= 0 && j2 <= src.cols);
        assert(i1 <= i2 && j1 <= j2);
        assert(rows == i2 - i1 && cols == j2 - j1);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                (*this)(i, j) = src(i1 + i, j1 + j);
            }
        }
    }

    void inplace_copy_column(float* dst, int j)
    {
        assert(j >= 0 && j < cols);
        for (int i = 0; i < rows; i++) {
            dst[i] = (*this)(i, j);
        }
    }

    void outplace_copy_row(float* dst, int i) const
    {
        assert(i >= 0 && i < rows);
        for (int j = 0; j < cols; j++) {
            dst[j] = (*this)(i, j);
        }
    }

    void outplace_copy_column(float* dst, int j) const
    {
        assert(j >= 0 && j < cols);
        for (int i = 0; i < rows; i++) {
            dst[i] = (*this)(i, j);
        }
    }

    void inplace_copy(const matview_2d& src)
    {
        assert(rows * cols == src.rows * src.cols);
        size_t nbytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
        std::memmove(data, src.data, nbytes);
    }

    void gemm(float alpha, const matview_2d& A, const matview_2d& B, float beta = 1.0f)
    {
        assert(rows == A.rows && cols == B.cols && A.cols == B.rows);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            rows, cols, A.cols,
            alpha, A.data, A.cols,
            B.data, B.cols,
            beta, data, cols);
    }

    void ger(float alpha, const float* x, int incx, const float* y, int incy)
    {
        cblas_sger(CblasRowMajor, rows, cols,
            alpha, x, incx, y, incy,
            data, cols);
    }
};

struct owned_mat_2d {
    std::unique_ptr<float[]> _data;
    int rows;
    int cols;

    owned_mat_2d(int rows, int cols)
        : _data(std::make_unique<float[]>(rows * cols))
        , rows(rows)
        , cols(cols)
    {
    }

    matview_2d view()
    {
        return matview_2d(_data.get(), rows, cols);
    }
};

void unweighted_kmeans_assign(int* assignments, const matview_2d& centroids, const matview_2d& points);
owned_mat_2d weighted_kmeans_train(const matview_2d& data, const float* weights, int k, int niter);
