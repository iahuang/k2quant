#include <algorithm>
#include <atomic>
#include <cblas.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>
#include <thread>
#include <vector>

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

    // this += alpha * A * B
    // this is (M x N), A is (M x K), B is (K x N)
    void gemm(float alpha, const matview_2d& A, const matview_2d& B, float beta = 1.0f)
    {
        assert(rows == A.rows && cols == B.cols && A.cols == B.rows);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            rows, cols, A.cols,
            alpha, A.data, A.cols,
            B.data, B.cols,
            beta, data, cols);
    }

    // this += alpha * x * y^T  (rank-1 update)
    // x has length rows, y has length cols
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

template <int D>
float square_l2_exact_d(const float* a, const float* b)
{
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

template <int D>
void unweighted_kmeans_assign_exact_d(int* assignments, const matview_2d& centroids, const matview_2d& points)
{
    int K = centroids.rows;
    int N = points.rows;
    assert(centroids.cols == D);
    assert(points.cols == D);

    for (int n = 0; n < N; n++) {
        float min_cost = std::numeric_limits<float>::infinity();
        int best_i = 0;
        const float* point = points[n];

        for (int i = 0; i < K; i++) {
            float cost = square_l2_exact_d<D>(point, centroids[i]);
            if (cost < min_cost) {
                min_cost = cost;
                best_i = i;
            }
        }

        assignments[n] = best_i;
    }
}

void unweighted_kmeans_assign_generic(int* assignments, const matview_2d& centroids, const matview_2d& points)
{
    int K = centroids.rows;
    int N = points.rows;
    int D = points.cols;

    for (int n = 0; n < N; n++) {
        float min_cost = std::numeric_limits<float>::infinity();
        int best_i = 0;
        const float* point = points[n];

        for (int i = 0; i < K; i++) {
            float cost = square_l2(point, centroids[i], D);
            if (cost < min_cost) {
                min_cost = cost;
                best_i = i;
            }
        }

        assignments[n] = best_i;
    }
}

void unweighted_kmeans_assign(int* assignments, const matview_2d& centroids, const matview_2d& points)
{
    int D = points.cols;
    switch (D) {
    case 1:
        return unweighted_kmeans_assign_exact_d<1>(assignments, centroids, points);
    case 2:
        return unweighted_kmeans_assign_exact_d<2>(assignments, centroids, points);
    case 3:
        return unweighted_kmeans_assign_exact_d<3>(assignments, centroids, points);
    case 4:
        return unweighted_kmeans_assign_exact_d<4>(assignments, centroids, points);
    case 5:
        return unweighted_kmeans_assign_exact_d<5>(assignments, centroids, points);
    case 6:
        return unweighted_kmeans_assign_exact_d<6>(assignments, centroids, points);
    case 7:
        return unweighted_kmeans_assign_exact_d<7>(assignments, centroids, points);
    case 8:
        return unweighted_kmeans_assign_exact_d<8>(assignments, centroids, points);
    case 9:
        return unweighted_kmeans_assign_exact_d<9>(assignments, centroids, points);
    case 10:
        return unweighted_kmeans_assign_exact_d<10>(assignments, centroids, points);
    case 11:
        return unweighted_kmeans_assign_exact_d<11>(assignments, centroids, points);
    case 12:
        return unweighted_kmeans_assign_exact_d<12>(assignments, centroids, points);
    case 13:
        return unweighted_kmeans_assign_exact_d<13>(assignments, centroids, points);
    case 14:
        return unweighted_kmeans_assign_exact_d<14>(assignments, centroids, points);
    case 15:
        return unweighted_kmeans_assign_exact_d<15>(assignments, centroids, points);
    case 16:
        return unweighted_kmeans_assign_exact_d<16>(assignments, centroids, points);
    default:
        break;
    }

    return unweighted_kmeans_assign_generic(assignments, centroids, points);
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

// Transposed centroid storage: (V, K) layout so that sweeps across K are contiguous
struct centroids_transposed {
    std::unique_ptr<float[]> data; // V * K floats
    int V;
    int K;

    centroids_transposed(const matview_2d& centroids)
        : data(std::make_unique<float[]>(centroids.cols * centroids.rows))
        , V(centroids.cols)
        , K(centroids.rows)
    {
        for (int k = 0; k < K; k++) {
            for (int v = 0; v < V; v++) {
                data[v * K + k] = centroids(k, v);
            }
        }
    }

    const float* component(int v) const { return data.get() + v * K; }
};

#include <immintrin.h>

// AVX2 vectorized argmin over K values. K must be a multiple of 8.
static int argmin_avx2(const float* dist, int K)
{
    __m256 min_vals = _mm256_loadu_ps(dist);
    __m256i min_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i step = _mm256_set1_epi32(8);
    __m256i cur_idx = min_idx;

    for (int k = 8; k < K; k += 8) {
        cur_idx = _mm256_add_epi32(cur_idx, step);
        __m256 vals = _mm256_loadu_ps(dist + k);
        __m256 mask = _mm256_cmp_ps(vals, min_vals, _CMP_LT_OS);
        min_vals = _mm256_blendv_ps(min_vals, vals, mask);
        min_idx = _mm256_blendv_epi8(min_idx, cur_idx,
                                      _mm256_castps_si256(mask));
    }

    // horizontal reduction: 8 → 1
    // swap high/low 128-bit lanes
    __m128 lo = _mm256_castps256_ps128(min_vals);
    __m128 hi = _mm256_extractf128_ps(min_vals, 1);
    __m128i ilo = _mm256_castsi256_si128(min_idx);
    __m128i ihi = _mm256_extracti128_si256(min_idx, 1);

    __m128 mask4 = _mm_cmplt_ps(hi, lo);
    lo = _mm_blendv_ps(lo, hi, mask4);
    ilo = _mm_blendv_epi8(ilo, ihi, _mm_castps_si128(mask4));

    // 4 → 2
    __m128 shuf = _mm_movehdup_ps(lo);           // [1,1,3,3]
    __m128i ishuf = _mm_shuffle_epi32(ilo, 0xB1); // [1,0,3,2]
    __m128 mask2 = _mm_cmplt_ps(shuf, lo);
    lo = _mm_blendv_ps(lo, shuf, mask2);
    ilo = _mm_blendv_epi8(ilo, ishuf, _mm_castps_si128(mask2));

    // 2 → 1
    __m128 shuf2 = _mm_movehl_ps(lo, lo);         // [2,3,2,3]
    __m128i ishuf2 = _mm_shuffle_epi32(ilo, 0x0E); // [2,3,_,_]
    __m128 mask1 = _mm_cmplt_ps(shuf2, lo);
    ilo = _mm_blendv_epi8(ilo, ishuf2, _mm_castps_si128(mask1));

    return _mm_cvtsi128_si32(ilo);
}

void unweighted_kmeans_assign_transposed(
    int* assignments,
    const centroids_transposed& ct,
    const matview_2d& points,
    float* dist_buf) // preallocated, length K
{
    int N = points.rows;
    int K = ct.K;
    int V = ct.V;

    for (int n = 0; n < N; n++) {
        const float* x = points[n];

        // first component: dist[k] = (x[0] - c0[k])^2
        {
            const float* c = ct.component(0);
            float xv = x[0];
            for (int k = 0; k < K; k++) {
                float diff = xv - c[k];
                dist_buf[k] = diff * diff;
            }
        }

        // remaining components: dist[k] += (x[v] - cv[k])^2
        for (int v = 1; v < V; v++) {
            const float* c = ct.component(v);
            float xv = x[v];
            for (int k = 0; k < K; k++) {
                float diff = xv - c[k];
                dist_buf[k] += diff * diff;
            }
        }

        // argmin

        assignments[n] = argmin_avx2(dist_buf, K);
    }
}

// Per-point weighted k-means. weights is a flat array of length N (one
// positive weight per training point). Assignment is unweighted (per-point
// scalars don't change the argmin); only the centroid update is weighted.
owned_mat_2d weighted_kmeans_train(const matview_2d& data, const float* weights, int k, int niter)
{
    int D = data.cols;
    int N = data.rows;
    constexpr float kmeans_centroid_delta_tol_sq = 1e-6f;

    auto centroids = owned_mat_2d(k, D);
    auto centroids_next = owned_mat_2d(k, D);
    auto weights_sum = std::make_unique<float[]>(k);
    auto assignments = std::make_unique<int[]>(N);
    auto dist_buf = std::make_unique<float[]>(k);

    auto cv = centroids.view();
    kmeans_centroid_init(data, cv, k);

    centroids_transposed ct(cv);

    for (int _ = 0; _ < niter; _++) {
        cv = centroids.view();
        unweighted_kmeans_assign_transposed(assignments.get(), ct, data, dist_buf.get());
        auto centroids_next_view = centroids_next.view();

        centroids_next_view.inplace_zero();
        std::fill(weights_sum.get(), weights_sum.get() + k, 0.0f);

        for (int i = 0; i < N; i++) {
            int assignment = assignments[i];
            float weight = weights[i];
            weights_sum[assignment] += weight;
            for (int d = 0; d < D; d++) {
                centroids_next_view(assignment, d) += data(i, d) * weight;
            }
        }

        for (int c = 0; c < k; c++) {
            float ws = weights_sum[c];
            if (ws > 0.0f) {
                float inv_ws = 1.0f / ws;
                for (int d = 0; d < D; d++) {
                    centroids_next_view(c, d) *= inv_ws;
                }
            } else {
                // Keep the previous centroid when a cluster receives no points.
                for (int d = 0; d < D; d++) {
                    centroids_next_view(c, d) = cv(c, d);
                }
            }
        }

        float max_centroid_delta_sq = 0.0f;
        for (int c = 0; c < k; c++) {
            max_centroid_delta_sq = std::max(
                max_centroid_delta_sq,
                square_l2(centroids_next_view[c], cv[c], D));
        }

        std::swap(centroids, centroids_next);

        ct = centroids_transposed(centroids.view());
        
        if (max_centroid_delta_sq <= kmeans_centroid_delta_tol_sq) {
            fprintf(stderr, "[weighted_kmeans_train] converged early after %d iterations\n", _);
            break;
        }
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
    constexpr int kVptqNumThreads = 48;

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
        int num_threads = std::min(kVptqNumThreads, n_experts);
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

PYBIND11_MODULE(vptq, m)
{
    m.doc() = "VPTQ quantization kernel";

    m.def("vptq_quantize", &vptq_quantize, "Quantize a weight matrix using VPTQ");
}