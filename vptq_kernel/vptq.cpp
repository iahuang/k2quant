#include "kmeans.h"

#include <algorithm>
#include <atomic>
#include <cblas.h>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <thread>
#include <utility>
#include <vector>

using hrc = std::chrono::high_resolution_clock;

static double elapsed_ms(hrc::time_point start)
{
    return std::chrono::duration<double, std::milli>(hrc::now() - start).count();
}

namespace py = pybind11;

py::array_t<float, py::array::c_style>
slice_axis0_3d(const py::array_t<float, py::array::c_style>& src, int i)
{
    int rows = src.shape(1);
    int cols = src.shape(2);

    py::array_t<float, py::array::c_style> dst({ rows, cols });

    auto uc_src = src.unchecked<3>();
    auto uc_mut_dst = dst.mutable_unchecked<2>();

    for (int k = 0; k < rows; k++) {
        for (int j = 0; j < cols; j++) {
            uc_mut_dst(k, j) = uc_src(i, k, j);
        }
    }

    return dst;
}

// returns (indices, centroids)
std::pair<std::vector<int>, owned_mat_2d> _vptq_quantize_one_expert(
    int ei,
    // assumed to be a copy that can be mutated in-place
    matview_2d& W_expert_quant,
    const matview_2d& Hinv,
    const float* h_diag,
    int V,
    int K,
    int kmeans_niter,
    int block_size)
{
    auto t_expert_start = hrc::now();

    int oc = W_expert_quant.rows;
    int ic = W_expert_quant.cols;
    int n_row_subvecs = oc / V;
    int N = n_row_subvecs * ic;

    owned_mat_2d train_data(N, V);

    auto t0 = hrc::now();
    auto point_weights = std::make_unique<float[]>(N);
    for (int col = 0; col < ic; col++) {
        for (int g = 0; g < n_row_subvecs; g++) {
            for (int v = 0; v < V; v++) {
                train_data.view()(col * n_row_subvecs + g, v) = W_expert_quant(g * V + v, col);
            }
            point_weights[col * n_row_subvecs + g] = h_diag[col];
        }
    }
    fprintf(stderr, "[expert %d] subvec extraction: %.1f ms\n", ei, elapsed_ms(t0));

    t0 = hrc::now();
    auto centroids = weighted_kmeans_train(train_data.view(), point_weights.get(), K, kmeans_niter);
    fprintf(stderr, "[expert %d] kmeans training (K=%d, niter=%d, N=%d, V=%d): %.1f ms\n",
        ei, K, kmeans_niter, n_row_subvecs * ic, V, elapsed_ms(t0));

    t0 = hrc::now();
    std::vector<int> indices(n_row_subvecs * ic);

    owned_mat_2d W1(oc, block_size);
    owned_mat_2d Err1(oc, block_size);
    owned_mat_2d Hinv1(block_size, block_size);
    auto q_col = std::make_unique<float[]>(oc);

    std::unique_ptr<float[]> col_vec(new float[oc]);
    std::unique_ptr<int[]> assignments(new int[n_row_subvecs]);

    for (int i1 = 0; i1 < ic; i1 += block_size) {
        int i2 = std::min(i1 + block_size, ic);
        int count = i2 - i1;

        // realloc; this should only happen at the end of the last block, if at all.
        if (count < block_size) {
            W1 = owned_mat_2d(oc, count);
            Err1 = owned_mat_2d(oc, count);
            Hinv1 = owned_mat_2d(count, count);
        }

        W1.view().inplace_copy_subview(W_expert_quant, 0, oc, i1, i2);
        Err1.view().inplace_zero();
        Hinv1.view().inplace_copy_subview(Hinv, i1, i2, i1, i2);

        // process one column at a time
        for (int j = 0; j < count; j++) {
            W1.view().outplace_copy_column(col_vec.get(), j);
            matview_2d sv(col_vec.get(), n_row_subvecs, V); // does this work?

            unweighted_kmeans_assign(assignments.get(), centroids.view(), sv);

            // q_col = centroids[assignments].reshape(oc)
            for (int g = 0; g < n_row_subvecs; g++) {
                for (int v = 0; v < V; v++) {
                    q_col[g * V + v] = centroids.view()(assignments[g], v);
                }
            }

            // indices[:, i1 + j] = assignments
            for (int i = 0; i < n_row_subvecs; i++) {
                indices[i * ic + i1 + j] = assignments[i];
            }

            // err = (W1[:, j] - centroids[assignments]) / Hinv1[j, j]
            // Err1[:, j] = err
            for (int i = 0; i < oc; i++) {
                Err1.view()(i, j) = (W1.view()(i, j) - q_col[i]) / Hinv1.view()(j, j);
            }

            if (j + 1 < count) {
                // W1[:, j+1:] -= Err1[:, j] * Hinv1[j, j+1:]
                int remaining = count - j - 1;
                cblas_sger(CblasRowMajor, oc, remaining,
                    -1.0f,
                    &Err1.view()(0, j), count, // x = Err1 column j, stride = count
                    &Hinv1.view()(j, j + 1), 1, // y = Hinv1 row j from j+1 onward
                    &W1.view()(0, j + 1), count); // C = W1 from column j+1, lda = count
            }
        }

        if (i2 < ic) {
            // W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                oc, ic - i2, count,
                -1.0f, Err1.view().data, Err1.view().cols,
                &Hinv(i1, i2), Hinv.cols,
                1.0f, &W_expert_quant(0, i2), W_expert_quant.cols);
        }
    }

    fprintf(stderr, "[expert %d] error propagation (%d blocks): %.1f ms\n",
        ei, (ic + block_size - 1) / block_size, elapsed_ms(t0));
    fprintf(stderr, "[expert %d] total: %.1f ms\n", ei, elapsed_ms(t_expert_start));

    return std::make_pair(std::move(indices), std::move(centroids));
}

// Error propagation only - centroids are pre-computed externally (e.g. by FAISS).
// Returns indices vector of length n_row_subvecs * ic.
std::vector<int> _vptq_errprop_one_expert(
    int ei,
    matview_2d& W_expert_quant,
    const matview_2d& Hinv,
    const matview_2d& centroids,
    int V,
    int block_size)
{
    auto t0 = hrc::now();

    int oc = W_expert_quant.rows;
    int ic = W_expert_quant.cols;
    int n_row_subvecs = oc / V;

    std::vector<int> indices(n_row_subvecs * ic);

    owned_mat_2d W1(oc, block_size);
    owned_mat_2d Err1(oc, block_size);
    owned_mat_2d Hinv1(block_size, block_size);
    auto q_col = std::make_unique<float[]>(oc);

    std::unique_ptr<float[]> col_vec(new float[oc]);
    std::unique_ptr<int[]> assignments(new int[n_row_subvecs]);

    for (int i1 = 0; i1 < ic; i1 += block_size) {
        int i2 = std::min(i1 + block_size, ic);
        int count = i2 - i1;

        if (count < block_size) {
            W1 = owned_mat_2d(oc, count);
            Err1 = owned_mat_2d(oc, count);
            Hinv1 = owned_mat_2d(count, count);
        }

        W1.view().inplace_copy_subview(W_expert_quant, 0, oc, i1, i2);
        Err1.view().inplace_zero();
        Hinv1.view().inplace_copy_subview(Hinv, i1, i2, i1, i2);

        for (int j = 0; j < count; j++) {
            W1.view().outplace_copy_column(col_vec.get(), j);
            matview_2d sv(col_vec.get(), n_row_subvecs, V);

            unweighted_kmeans_assign(assignments.get(), centroids, sv);

            for (int g = 0; g < n_row_subvecs; g++) {
                for (int v = 0; v < V; v++) {
                    q_col[g * V + v] = centroids(assignments[g], v);
                }
            }

            for (int i = 0; i < n_row_subvecs; i++) {
                indices[i * ic + i1 + j] = assignments[i];
            }

            for (int i = 0; i < oc; i++) {
                Err1.view()(i, j) = (W1.view()(i, j) - q_col[i]) / Hinv1.view()(j, j);
            }

            if (j + 1 < count) {
                int remaining = count - j - 1;
                cblas_sger(CblasRowMajor, oc, remaining,
                    -1.0f,
                    &Err1.view()(0, j), count,
                    &Hinv1.view()(j, j + 1), 1,
                    &W1.view()(0, j + 1), count);
            }
        }

        if (i2 < ic) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                oc, ic - i2, count,
                -1.0f, Err1.view().data, Err1.view().cols,
                &Hinv(i1, i2), Hinv.cols,
                1.0f, &W_expert_quant(0, i2), W_expert_quant.cols);
        }
    }

    fprintf(stderr, "[expert %d] error propagation: %.1f ms\n", ei, elapsed_ms(t0));

    return indices;
}

// Hybrid entry point: accepts pre-computed centroids (from e.g. FAISS), runs
// only the error-propagation + assignment step in C++.
// W_quant:      (n_experts, oc, ic)   float32 C-contiguous
// Hinv:         (ic, ic)              float32 C-contiguous
// centroids_all:(n_experts * K, V)    float32 C-contiguous  (expert ei at row ei*K)
// Returns:      (n_experts, n_row_subvecs * ic)  int32
py::array_t<int, py::array::c_style>
vptq_errprop(
    py::array_t<float, py::array::c_style>& W_quant,
    py::array_t<float, py::array::c_style>& Hinv,
    py::array_t<float, py::array::c_style>& centroids_all,
    int V,
    int K,
    int block_size)
{
    auto t_total = hrc::now();
    constexpr int vptq_num_threads = 24;

    assert(W_quant.ndim() == 3);
    int n_experts = W_quant.shape(0);
    int oc = W_quant.shape(1);
    int ic = W_quant.shape(2);
    int n_row_subvecs = oc / V;

    auto Hinv_data = matview_2d::from_array_f32(Hinv);

    assert(centroids_all.ndim() == 2);
    assert(centroids_all.shape(0) == n_experts * K);
    assert(centroids_all.shape(1) == V);
    float* centroids_ptr = centroids_all.mutable_data();

    fprintf(stderr, "[vptq_errprop] %d experts (V=%d, K=%d, block=%d)\n",
        n_experts, V, K, block_size);

    std::vector<py::array_t<float, py::array::c_style>> per_expert_weights;
    std::vector<matview_2d> per_expert_weight_views;
    per_expert_weights.reserve(n_experts);
    per_expert_weight_views.reserve(n_experts);

    for (int ei = 0; ei < n_experts; ei++) {
        per_expert_weights.push_back(slice_axis0_3d(W_quant, ei));
        per_expert_weight_views.push_back(matview_2d::from_array_f32(per_expert_weights.back()));
    }

    std::vector<std::unique_ptr<std::vector<int>>> per_expert_indices(n_experts);

    {
        py::gil_scoped_release release;
        int num_threads = std::min(vptq_num_threads, n_experts);
        std::atomic<int> next_ei(0);
        std::exception_ptr worker_error;
        std::mutex worker_error_mutex;
        std::vector<std::thread> workers;
        workers.reserve(num_threads);

        for (int ti = 0; ti < num_threads; ti++) {
            workers.emplace_back([&]() {
                try {
                    while (true) {
                        int ei = next_ei.fetch_add(1);
                        if (ei >= n_experts) break;

                        matview_2d expert_centroids(centroids_ptr + ei * K * V, K, V);
                        per_expert_indices[ei] = std::make_unique<std::vector<int>>(
                            _vptq_errprop_one_expert(
                                ei,
                                per_expert_weight_views[ei],
                                Hinv_data,
                                expert_centroids,
                                V, block_size));
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(worker_error_mutex);
                    if (!worker_error) {
                        worker_error = std::current_exception();
                    }
                }
            });
        }

        for (auto& w : workers) w.join();
        if (worker_error) std::rethrow_exception(worker_error);
    }

    py::array_t<int, py::array::c_style> indices({
        static_cast<ssize_t>(n_experts),
        static_cast<ssize_t>(n_row_subvecs * ic)
    });
    auto indices_view = indices.mutable_unchecked<2>();
    for (int ei = 0; ei < n_experts; ei++) {
        auto& result = *per_expert_indices[ei];
        for (size_t i = 0; i < result.size(); i++) {
            indices_view(ei, i) = result[i];
        }
    }

    fprintf(stderr, "[vptq_errprop] total: %.1f ms\n", elapsed_ms(t_total));
    return indices;
}

// assumes that W_quant is padded so that its # rows (oc) are divisible by V
// also assumes pre-ordering of columns.
py::tuple vptq_quantize(
    py::array_t<float, py::array::c_style>& W_quant,
    py::array_t<float, py::array::c_style>& Hinv,
    py::array_t<float, py::array::c_style>& h_diag,
    int V,
    int K,
    int kmeans_niter,
    int block_size)
{
    auto t_total = hrc::now();
    constexpr int vptq_num_threads = 24;

    assert(W_quant.ndim() == 3);
    int n_experts = W_quant.shape(0);
    auto Hinv_data = matview_2d::from_array_f32(Hinv);
    assert(h_diag.ndim() == 1);
    const float* h_diag_ptr = h_diag.data();

    fprintf(stderr, "[vptq] quantizing %d experts (V=%d, K=%d, niter=%d, block=%d)\n",
        n_experts, V, K, kmeans_niter, block_size);

    std::vector<py::array_t<float, py::array::c_style>> per_expert_weights;
    std::vector<matview_2d> per_expert_weight_views;
    per_expert_weights.reserve(n_experts);
    per_expert_weight_views.reserve(n_experts);

    for (int ei = 0; ei < n_experts; ei++) {
        per_expert_weights.push_back(slice_axis0_3d(W_quant, ei));
        per_expert_weight_views.push_back(matview_2d::from_array_f32(per_expert_weights.back()));
    }

    std::vector<std::unique_ptr<std::pair<std::vector<int>, owned_mat_2d>>> per_expert_results(n_experts);

    {
        py::gil_scoped_release release;
        int num_threads = std::min(vptq_num_threads, n_experts);
        std::atomic<int> next_ei(0);
        std::exception_ptr worker_error;
        std::mutex worker_error_mutex;
        std::vector<std::thread> workers;
        workers.reserve(num_threads);

        for (int ti = 0; ti < num_threads; ti++) {
            workers.emplace_back([&]() {
                try {
                    while (true) {
                        int ei = next_ei.fetch_add(1);
                        if (ei >= n_experts) {
                            break;
                        }

                        per_expert_results[ei] = std::make_unique<std::pair<std::vector<int>, owned_mat_2d>>(
                            _vptq_quantize_one_expert(
                                ei,
                                per_expert_weight_views[ei],
                                Hinv_data,
                                h_diag_ptr,
                                V,
                                K,
                                kmeans_niter,
                                block_size));
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(worker_error_mutex);
                    if (!worker_error) {
                        worker_error = std::current_exception();
                    }
                }
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }

        if (worker_error) {
            std::rethrow_exception(worker_error);
        }
    }

    assert(n_experts > 0);
    auto& first_result = *per_expert_results[0];

    py::array_t<int, py::array::c_style> indices({ static_cast<ssize_t>(n_experts),
        static_cast<ssize_t>(first_result.first.size()) });

    py::array_t<float, py::array::c_style> codebooks({ static_cast<ssize_t>(n_experts) * static_cast<ssize_t>(first_result.second.rows),
        static_cast<ssize_t>(first_result.second.cols) });

    auto indices_view = indices.mutable_unchecked<2>();
    auto codebooks_view = codebooks.mutable_unchecked<2>();

    // copy results out
    for (int ei = 0; ei < n_experts; ei++) {
        auto& result = *per_expert_results[ei];

        for (size_t i = 0; i < result.first.size(); i++) {
            indices_view(ei, i) = result.first[i];
        }
    }

    for (int ei = 0; ei < n_experts; ei++) {
        auto& result = *per_expert_results[ei];
        for (int i = 0; i < result.second.rows; i++) {
            for (int j = 0; j < result.second.cols; j++) {
                codebooks_view(ei * result.second.rows + i, j) = result.second.view()(i, j);
            }
        }
    }

    fprintf(stderr, "[vptq] total quantization: %.1f ms\n", elapsed_ms(t_total));

    return py::make_tuple(indices, codebooks);
}

// KMeans++ initialization wrapper.
// data: (N, D) float32 C-contiguous
// k: number of centroids
// Returns: (k, D) float32 C-contiguous
py::array_t<float, py::array::c_style>
kmeanspp_init_py(
    py::array_t<float, py::array::c_style>& data,
    int k)
{
    assert(data.ndim() == 2);
    auto data_view = matview_2d::from_array_f32(data);
    int D = data.shape(1);

    owned_mat_2d centroids(0, 0);
    {
        py::gil_scoped_release release;
        centroids = kmeanspp_init(data_view, k);
    }

    py::array_t<float, py::array::c_style> result({ static_cast<ssize_t>(k), static_cast<ssize_t>(D) });
    auto result_view = result.mutable_unchecked<2>();
    auto cv = centroids.view();
    for (int i = 0; i < k; i++) {
        for (int d = 0; d < D; d++) {
            result_view(i, d) = cv(i, d);
        }
    }
    return result;
}

PYBIND11_MODULE(vptq, m)
{
    m.doc() = "VPTQ quantization kernel";

    m.def("vptq_quantize", &vptq_quantize, "Quantize a weight matrix using VPTQ");
    m.def("vptq_errprop", &vptq_errprop, "Error propagation with pre-computed centroids");
    m.def("kmeanspp_init", &kmeanspp_init_py, "KMeans++ initialization");
}