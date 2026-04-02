#include "kmeans.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace {

float square_l2(const float* a, const float* b, int d)
{
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
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

template <int D>
void kmeans_centroid_init_exact_d(const matview_2d& data, matview_2d& centroids, int k)
{
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
        const float* last_centroid = centroids[c - 1];
        for (int i = 0; i < N; i++) {
            float d2 = square_l2_exact_d<D>(data[i], last_centroid);
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

void kmeans_centroid_init(const matview_2d& data, matview_2d& centroids, int k)
{
    switch (data.cols) {
    case 1: return kmeans_centroid_init_exact_d<1>(data, centroids, k);
    case 2: return kmeans_centroid_init_exact_d<2>(data, centroids, k);
    case 3: return kmeans_centroid_init_exact_d<3>(data, centroids, k);
    case 4: return kmeans_centroid_init_exact_d<4>(data, centroids, k);
    case 8: return kmeans_centroid_init_exact_d<8>(data, centroids, k);
    case 16: return kmeans_centroid_init_exact_d<16>(data, centroids, k);
    case 32: return kmeans_centroid_init_exact_d<32>(data, centroids, k);
    case 64: return kmeans_centroid_init_exact_d<64>(data, centroids, k);
    default: break;
    }

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
        const float* last_centroid = centroids[c - 1];
        for (int i = 0; i < N; i++) {
            float d2 = square_l2(data[i], last_centroid, D);
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

struct centroids_transposed {
    std::unique_ptr<float[]> data;
    int V;
    int K;

    explicit centroids_transposed(const matview_2d& centroids)
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

    const float* component(int v) const
    {
        return data.get() + v * K;
    }
};

void unweighted_kmeans_assign_transposed(
    int* __restrict__ assignments,
    const centroids_transposed& ct,
    const matview_2d& points,
    float* __restrict__ dist_buf)
{
    int N = points.rows;
    int K = ct.K;
    int V = ct.V;
    constexpr int BATCH = 4;

    int n = 0;
    for (; n + BATCH <= N; n += BATCH) {
        float* d0 = dist_buf;
        float* d1 = dist_buf + K;
        float* d2 = dist_buf + 2 * K;
        float* d3 = dist_buf + 3 * K;
        const float* x0 = points[n];
        const float* x1 = points[n + 1];
        const float* x2 = points[n + 2];
        const float* x3 = points[n + 3];

        {
            const float* c = ct.component(0);
            float xv0 = x0[0], xv1 = x1[0], xv2 = x2[0], xv3 = x3[0];
            for (int k = 0; k < K; k++) {
                float cv = c[k];
                float diff0 = xv0 - cv;
                float diff1 = xv1 - cv;
                float diff2 = xv2 - cv;
                float diff3 = xv3 - cv;
                d0[k] = diff0 * diff0;
                d1[k] = diff1 * diff1;
                d2[k] = diff2 * diff2;
                d3[k] = diff3 * diff3;
            }
        }

        for (int v = 1; v < V; v++) {
            const float* c = ct.component(v);
            float xv0 = x0[v], xv1 = x1[v], xv2 = x2[v], xv3 = x3[v];
            for (int k = 0; k < K; k++) {
                float cv = c[k];
                float diff0 = xv0 - cv;
                float diff1 = xv1 - cv;
                float diff2 = xv2 - cv;
                float diff3 = xv3 - cv;
                d0[k] += diff0 * diff0;
                d1[k] += diff1 * diff1;
                d2[k] += diff2 * diff2;
                d3[k] += diff3 * diff3;
            }
        }

        for (int b = 0; b < BATCH; b++) {
            float* d = dist_buf + b * K;
            float best = d[0];
            int best_k = 0;
            for (int k = 1; k < K; k++) {
                if (d[k] < best) {
                    best = d[k];
                    best_k = k;
                }
            }
            assignments[n + b] = best_k;
        }
    }

    for (; n < N; n++) {
        const float* x = points[n];
        float* d = dist_buf;
        {
            const float* c = ct.component(0);
            float xv = x[0];
            for (int k = 0; k < K; k++) {
                float diff = xv - c[k];
                d[k] = diff * diff;
            }
        }
        for (int v = 1; v < V; v++) {
            const float* c = ct.component(v);
            float xv = x[v];
            for (int k = 0; k < K; k++) {
                float diff = xv - c[k];
                d[k] += diff * diff;
            }
        }
        float best = d[0];
        int best_k = 0;
        for (int k = 1; k < K; k++) {
            if (d[k] < best) {
                best = d[k];
                best_k = k;
            }
        }
        assignments[n] = best_k;
    }
}

} // namespace

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
    case 8:
        return unweighted_kmeans_assign_exact_d<8>(assignments, centroids, points);
    case 16:
        return unweighted_kmeans_assign_exact_d<16>(assignments, centroids, points);
    case 32:
        return unweighted_kmeans_assign_exact_d<32>(assignments, centroids, points);
    case 64:
        return unweighted_kmeans_assign_exact_d<64>(assignments, centroids, points);
    default:
        break;
    }

    return unweighted_kmeans_assign_generic(assignments, centroids, points);
}

owned_mat_2d weighted_kmeans_train(const matview_2d& data, const float* weights, int k, int niter)
{
    int D = data.cols;
    int N = data.rows;
    constexpr float kmeans_centroid_delta_tol_sq = 1e-6f;

    auto centroids = owned_mat_2d(k, D);
    auto centroids_next = owned_mat_2d(k, D);
    auto weights_sum = std::make_unique<float[]>(k);
    auto assignments = std::make_unique<int[]>(N);
    constexpr int NUM_THREADS = 32;
    // Each thread needs its own dist_buf (k * BATCH floats)
    auto dist_bufs = std::make_unique<float[]>(k * 4 * NUM_THREADS);

    auto cv = centroids.view();
    kmeans_centroid_init(data, cv, k);

    centroids_transposed ct(cv);

    for (int iter = 0; iter < niter; iter++) {
        cv = centroids.view();

        // Parallel assignment across NUM_THREADS threads
        {
            std::vector<std::thread> threads;
            threads.reserve(NUM_THREADS);
            int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;
            for (int t = 0; t < NUM_THREADS; t++) {
                int start = t * chunk;
                int end = std::min(start + chunk, N);
                if (start >= end) break;
                threads.emplace_back([&, start, end, t]() {
                    matview_2d points_slice(const_cast<float*>(data[start]), end - start, D);
                    float* my_dist_buf = dist_bufs.get() + t * k * 4;
                    unweighted_kmeans_assign_transposed(
                        assignments.get() + start, ct, points_slice, my_dist_buf);
                });
            }
            for (auto& th : threads) th.join();
        }
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
            fprintf(stderr, "[weighted_kmeans_train] converged early after %d iterations\n", iter);
            break;
        }
    }

    return centroids;
}
