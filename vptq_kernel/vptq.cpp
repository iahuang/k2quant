#include <cblas.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>

using hrc = std::chrono::high_resolution_clock;

static double elapsed_ms(hrc::time_point start)
{
    return std::chrono::duration<double, std::milli>(hrc::now() - start).count();
}

namespace py = pybind11;

// basic row-major 2d mutable matrix view type. does not own data. make sure
// that this object doesn't outlive its pointer!
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

    static matview_2d from_array_f32(py::array_t<float, py::array::c_style>& src)
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

    // copy a subview of src in place
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

    // copy the data of src into this matrix, assuming src has the same area
    // but not necessarily the same shape
    void inplace_copy(const matview_2d& src)
    {
        assert(rows * cols == src.rows * src.cols);
        size_t nbytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
        std::memmove(data, src.data, nbytes);
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

// maybe vectorize somehow
float square_l2(const float* a, const float* b, int d)
{
    float sum = 0.0;
    for (int i = 0; i < d; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

std::unique_ptr<int[]> unweighted_kmeans_assign(const matview_2d& centroids, const matview_2d& points)
{
    int K = centroids.rows;
    int N = points.rows;
    int D = points.cols;

    std::unique_ptr<int[]> assignments(new int[N]);

    for (int n = 0; n < N; n++) {
        float min_cost = INFINITY;

        for (int i = 0; i < K; i++) {
            float cost = square_l2(points[n], centroids[i], D);
            if (cost < min_cost) {
                min_cost = cost;
                assignments[n] = i;
            }
        }
    }

    return assignments;
}

void kmeans_centroid_init(const matview_2d& data, matview_2d& centroids, int k)
{
    int D = data.cols;
    int N = data.rows;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> first_dist(0, N - 1);

    int first_idx = first_dist(rng);
    for (int d = 0; d < D; d++) {
        centroids(0, d) = data(first_idx, d);
    }

    auto min_dists = std::make_unique<float[]>(N);

    for (int c = 1; c < k; c++) {
        float total = 0.0f;
        for (int i = 0; i < N; i++) {
            float d2 = square_l2(data[i], centroids[c - 1], D);
            min_dists[i] = (c == 1) ? d2 : std::min(min_dists[i], d2);
            total += min_dists[i];
        }

        if (total <= 0.0f) {
            for (int d = 0; d < D; d++) {
                centroids(c, d) = data(first_dist(rng), d);
            }
            continue;
        }

        std::uniform_real_distribution<float> uniform(0.0f, total);
        float r = uniform(rng);
        float cumsum = 0.0f;
        int selected = N - 1;
        for (int i = 0; i < N; i++) {
            cumsum += min_dists[i];
            if (cumsum >= r) {
                selected = i;
                break;
            }
        }

        for (int d = 0; d < D; d++) {
            centroids(c, d) = data(selected, d);
        }
    }
}

// Per-point weighted k-means. weights is a flat array of length N (one
// positive weight per training point). Assignment is unweighted (per-point
// scalars don't change the argmin); only the centroid update is weighted.
owned_mat_2d weighted_kmeans_train(const matview_2d& data, const float* weights, int k, int niter)
{
    int D = data.cols;
    int N = data.rows;

    auto centroids = owned_mat_2d(k, D);
    auto centroids_next = owned_mat_2d(k, D);
    auto weights_sum = std::make_unique<float[]>(k);

    auto cv = centroids.view();
    kmeans_centroid_init(data, cv, k);

    for (int _ = 0; _ < niter; _++) {
        cv = centroids.view();
        auto assignments = unweighted_kmeans_assign(cv, data);

        centroids_next.view().inplace_zero();
        std::fill(weights_sum.get(), weights_sum.get() + k, 0.0f);

        for (int i = 0; i < N; i++) {
            weights_sum[assignments[i]] += weights[i];
        }

        for (int i = 0; i < N; i++) {
            float ws = weights_sum[assignments[i]];
            for (int d = 0; d < D; d++) {
                centroids_next.view()(assignments[i], d) += data(i, d) * weights[i] / ws;
            }
        }

        std::swap(centroids, centroids_next);
    }

    return centroids;
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

            auto assignments = unweighted_kmeans_assign(centroids.view(), sv);

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
                // W1[:, j + 1 :] -= Err1[:, j] * Hinv1[j, j + 1 :]
                for (int i = 0; i < oc; i++) {
                    for (int k = j + 1; k < count; k++) {
                        W1.view()(i, k) -= Err1.view()(i, j) * Hinv1.view()(j, k);
                    }
                }
            }
        }

        if (i2 < ic) {
            // W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
            for (int i = 0; i < oc; i++) {
                for (int m = 0; m < ic - i2; m++) {
                    float sum = 0.0f;
                    for (int k = 0; k < count; k++) {
                        sum += Err1.view()(i, k) * Hinv(i1 + k, i2 + m);
                    }
                    W_expert_quant(i, i2 + m) -= sum;
                }
            }
        }
    }

    fprintf(stderr, "[expert %d] error propagation (%d blocks): %.1f ms\n",
        ei, (ic + block_size - 1) / block_size, elapsed_ms(t0));
    fprintf(stderr, "[expert %d] total: %.1f ms\n", ei, elapsed_ms(t_expert_start));

    return std::make_pair(std::move(indices), std::move(centroids));
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

    assert(W_quant.ndim() == 3);
    int n_experts = W_quant.shape(0);
    auto Hinv_data = matview_2d::from_array_f32(Hinv);
    assert(h_diag.ndim() == 1);
    const float* h_diag_ptr = h_diag.data();

    fprintf(stderr, "[vptq] quantizing %d experts (V=%d, K=%d, niter=%d, block=%d)\n",
        n_experts, V, K, kmeans_niter, block_size);

    std::vector<py::array_t<float, py::array::c_style>> per_expert_weights;

    for (int ei = 0; ei < n_experts; ei++) {
        per_expert_weights.push_back(slice_axis0_3d(W_quant, ei));
    }

    std::vector<std::pair<std::vector<int>, owned_mat_2d>> per_expert_results;

    // todo: parallelize
    {
        py::gil_scoped_release release;

        for (int ei = 0; ei < n_experts; ei++) {
            auto W_expert_quant = matview_2d::from_array_f32(per_expert_weights[ei]);
            auto result = _vptq_quantize_one_expert(ei, W_expert_quant, Hinv_data, h_diag_ptr, V, K, kmeans_niter, block_size);
            per_expert_results.push_back(std::move(result));
        }
    }

    assert(n_experts > 0);
    auto& first_result = per_expert_results[0];

    py::array_t<int, py::array::c_style> indices({ static_cast<ssize_t>(n_experts),
        static_cast<ssize_t>(first_result.first.size()) });

    py::array_t<float, py::array::c_style> codebooks({ static_cast<ssize_t>(n_experts) * static_cast<ssize_t>(first_result.second.rows),
        static_cast<ssize_t>(first_result.second.cols) });

    auto indices_view = indices.mutable_unchecked<2>();
    auto codebooks_view = codebooks.mutable_unchecked<2>();

    // copy results out
    for (int ei = 0; ei < n_experts; ei++) {
        auto& result = per_expert_results[ei];

        for (int i = 0; i < result.first.size(); i++) {
            indices_view(ei, i) = result.first[i];
        }
    }

    for (int ei = 0; ei < n_experts; ei++) {
        auto& result = per_expert_results[ei];
        for (int i = 0; i < result.second.rows; i++) {
            for (int j = 0; j < result.second.cols; j++) {
                codebooks_view(ei * result.second.rows + i, j) = result.second.view()(i, j);
            }
        }
    }

    fprintf(stderr, "[vptq] total quantization: %.1f ms\n", elapsed_ms(t_total));

    return py::make_tuple(indices, codebooks);
}

PYBIND11_MODULE(vptq, m)
{
    m.doc() = "VPTQ quantization kernel";

    m.def("vptq_quantize", &vptq_quantize, "Quantize a weight matrix using VPTQ");
}