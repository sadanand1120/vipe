#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tsdf_ext {

struct Voxel {
    float tsdf = 0.0f;
    float weight = 0.0f;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

struct BlockKey {
    int x;
    int y;
    int z;

    bool operator==(const BlockKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey &key) const {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](int v) {
            h ^= static_cast<uint32_t>(v);
            h *= 1099511628211ull;
        };
        mix(key.x);
        mix(key.y);
        mix(key.z);
        return static_cast<size_t>(h);
    }
};

struct SurfaceVertex {
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
    float nx;
    float ny;
    float nz;
};

struct SurfaceTriangle {
    SurfaceVertex a;
    SurfaceVertex b;
    SurfaceVertex c;
    double area;
};

#pragma pack(push, 1)
struct PlyVertex {
    float x;
    float y;
    float z;
    float nx;
    float ny;
    float nz;
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t normals_red;
    uint8_t normals_green;
    uint8_t normals_blue;
};
#pragma pack(pop)

static_assert(sizeof(PlyVertex) == 30, "PLY vertex layout must match the Python structured dtype");

static constexpr size_t kPlyChunkVertices = 65536;

struct Representative {
    PlyVertex vertex;
    float distance_sq;
};

static uint8_t clamp_ply_color(float value) {
    value = std::min(255.0f, std::max(0.0f, value));
    return static_cast<uint8_t>(std::round(value));
}

static void write_ply_header(std::ofstream &ply_file, int64_t vertex_count) {
    ply_file << "ply\n"
             << "format binary_little_endian 1.0\n"
             << "element vertex " << vertex_count << "\n"
             << "property float x\n"
             << "property float y\n"
             << "property float z\n"
             << "property float nx\n"
             << "property float ny\n"
             << "property float nz\n"
             << "property uchar red\n"
             << "property uchar green\n"
             << "property uchar blue\n"
             << "property uchar normals_red\n"
             << "property uchar normals_green\n"
             << "property uchar normals_blue\n"
             << "end_header\n";
}

struct Matrix4f {
    float v[16];

    float operator()(int r, int c) const {
        return v[r * 4 + c];
    }
};

struct Intrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
};

struct Vec3f {
    float x;
    float y;
    float z;
};

static int floor_div(int a, int b) {
    int q = a / b;
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) {
        --q;
    }
    return q;
}

static int floor_mod(int a, int b) {
    return a - floor_div(a, b) * b;
}

static int block_index_from_world(float coord_m, float block_edge_m) {
    return static_cast<int>(std::floor(coord_m / block_edge_m));
}

static Matrix4f tensor_to_matrix4(torch::Tensor tensor) {
    tensor = tensor.contiguous();
    TORCH_CHECK(tensor.device().is_cpu(), "extrinsic must be a CPU tensor");
    TORCH_CHECK(tensor.numel() == 16, "extrinsic must have 16 values");
    Matrix4f out;
    if (tensor.scalar_type() == at::ScalarType::Double) {
        const double *ptr = tensor.data_ptr<double>();
        for (int i = 0; i < 16; ++i) out.v[i] = static_cast<float>(ptr[i]);
    } else {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float, "extrinsic must be float32 or float64");
        const float *ptr = tensor.data_ptr<float>();
        for (int i = 0; i < 16; ++i) out.v[i] = ptr[i];
    }
    return out;
}

static Intrinsics tensor_to_intrinsics(torch::Tensor tensor) {
    tensor = tensor.contiguous();
    TORCH_CHECK(tensor.device().is_cpu(), "intrinsics must be a CPU tensor");
    Intrinsics intr;
    if (tensor.numel() == 4) {
        if (tensor.scalar_type() == at::ScalarType::Double) {
            const double *ptr = tensor.data_ptr<double>();
            intr = {static_cast<float>(ptr[0]), static_cast<float>(ptr[1]), static_cast<float>(ptr[2]),
                    static_cast<float>(ptr[3])};
        } else {
            TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float, "intrinsics must be float32 or float64");
            const float *ptr = tensor.data_ptr<float>();
            intr = {ptr[0], ptr[1], ptr[2], ptr[3]};
        }
    } else {
        TORCH_CHECK(tensor.numel() == 9, "intrinsics must have 4 or 9 values");
        if (tensor.scalar_type() == at::ScalarType::Double) {
            const double *ptr = tensor.data_ptr<double>();
            intr = {static_cast<float>(ptr[0]), static_cast<float>(ptr[4]), static_cast<float>(ptr[2]),
                    static_cast<float>(ptr[5])};
        } else {
            TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float, "intrinsics must be float32 or float64");
            const float *ptr = tensor.data_ptr<float>();
            intr = {ptr[0], ptr[4], ptr[2], ptr[5]};
        }
    }
    return intr;
}

static Matrix4f invert_rigid_w2c(const Matrix4f &w2c) {
    Matrix4f c2w{};
    c2w.v[15] = 1.0f;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            c2w.v[r * 4 + c] = w2c(c, r);
        }
    }
    for (int r = 0; r < 3; ++r) {
        c2w.v[r * 4 + 3] =
                -(c2w(r, 0) * w2c(0, 3) + c2w(r, 1) * w2c(1, 3) + c2w(r, 2) * w2c(2, 3));
    }
    return c2w;
}

class TSDFVolume {
   public:
    TSDFVolume(double voxel_edge_m, double sdf_trunc_m, int num_voxels_per_block_edge, int depth_sampling_stride)
        : voxel_edge_m_(static_cast<float>(voxel_edge_m)),
          sdf_trunc_m_(static_cast<float>(sdf_trunc_m)),
          num_voxels_per_block_edge_(num_voxels_per_block_edge),
          depth_sampling_stride_(depth_sampling_stride) {
        TORCH_CHECK(voxel_edge_m_ > 0.0f, "voxel_edge_m must be positive");
        TORCH_CHECK(sdf_trunc_m_ > 0.0f, "sdf_trunc_m must be positive");
        TORCH_CHECK(num_voxels_per_block_edge_ > 0, "num_voxels_per_block_edge must be positive");
        TORCH_CHECK(depth_sampling_stride_ > 0, "depth_sampling_stride must be positive");
        num_voxels_per_block_ =
                num_voxels_per_block_edge_ * num_voxels_per_block_edge_ * num_voxels_per_block_edge_;
        block_edge_m_ = voxel_edge_m_ * static_cast<float>(num_voxels_per_block_edge_);
    }

    void integrate(torch::Tensor depth,
                   torch::Tensor color,
                   torch::Tensor intrinsics_tensor,
                   torch::Tensor extrinsic_tensor,
                   double depth_trunc) {
        depth = depth.contiguous();
        color = color.contiguous();
        TORCH_CHECK(depth.device().is_cpu(), "depth must be a CPU tensor");
        TORCH_CHECK(color.device().is_cpu(), "color must be a CPU tensor");
        TORCH_CHECK(depth.scalar_type() == at::ScalarType::Float, "depth must be float32");
        TORCH_CHECK(color.scalar_type() == at::ScalarType::Byte, "color must be uint8");
        TORCH_CHECK(depth.dim() == 2, "depth must have shape HxW");
        TORCH_CHECK(color.dim() == 3 && color.size(2) == 3, "color must have shape HxWx3");
        TORCH_CHECK(depth.size(0) == color.size(0) && depth.size(1) == color.size(1),
                    "depth and color shapes must match");

        const int height = static_cast<int>(depth.size(0));
        const int width = static_cast<int>(depth.size(1));
        const float *depth_ptr = depth.data_ptr<float>();
        const uint8_t *color_ptr = color.data_ptr<uint8_t>();
        const Intrinsics intr = tensor_to_intrinsics(intrinsics_tensor);
        const Matrix4f w2c = tensor_to_matrix4(extrinsic_tensor);
        const Matrix4f c2w = invert_rigid_w2c(w2c);
        const float depth_trunc_f = static_cast<float>(depth_trunc);

        std::unordered_set<BlockKey, BlockKeyHash> touched;
        for (int v = 0; v < height; v += depth_sampling_stride_) {
            for (int u = 0; u < width; u += depth_sampling_stride_) {
                const float d = depth_ptr[v * width + u];
                if (!std::isfinite(d) || d <= 0.0f || d > depth_trunc_f) {
                    continue;
                }
                const float x = (static_cast<float>(u) - intr.cx) * d / intr.fx;
                const float y = (static_cast<float>(v) - intr.cy) * d / intr.fy;
                const float z = d;
                const float wx = c2w(0, 0) * x + c2w(0, 1) * y + c2w(0, 2) * z + c2w(0, 3);
                const float wy = c2w(1, 0) * x + c2w(1, 1) * y + c2w(1, 2) * z + c2w(1, 3);
                const float wz = c2w(2, 0) * x + c2w(2, 1) * y + c2w(2, 2) * z + c2w(2, 3);
                const int min_x = block_index_from_world(wx - sdf_trunc_m_, block_edge_m_);
                const int min_y = block_index_from_world(wy - sdf_trunc_m_, block_edge_m_);
                const int min_z = block_index_from_world(wz - sdf_trunc_m_, block_edge_m_);
                const int max_x = block_index_from_world(wx + sdf_trunc_m_, block_edge_m_);
                const int max_y = block_index_from_world(wy + sdf_trunc_m_, block_edge_m_);
                const int max_z = block_index_from_world(wz + sdf_trunc_m_, block_edge_m_);
                for (int bx = min_x; bx <= max_x; ++bx) {
                    for (int by = min_y; by <= max_y; ++by) {
                        for (int bz = min_z; bz <= max_z; ++bz) {
                            touched.insert({bx, by, bz});
                        }
                    }
                }
            }
        }

        std::vector<BlockKey> keys(touched.begin(), touched.end());
        std::sort(keys.begin(), keys.end(), [](const BlockKey &a, const BlockKey &b) {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.z < b.z;
        });

        std::vector<std::vector<Voxel> *> touched_block_ptrs;
        touched_block_ptrs.reserve(keys.size());
        for (const auto &key : keys) {
            auto it = all_blocks_.find(key);
            if (it == all_blocks_.end()) {
                it = all_blocks_.emplace(key, std::vector<Voxel>(num_voxels_per_block_)).first;
            }
            touched_block_ptrs.push_back(&it->second);
        }

        const float safe_width = static_cast<float>(width) - 0.0001f;
        const float safe_height = static_cast<float>(height) - 0.0001f;
        const float sdf_trunc_m_inv = 1.0f / sdf_trunc_m_;

        at::parallel_for(0, static_cast<int64_t>(keys.size()), 1, [&](int64_t begin, int64_t end) {
            for (int64_t bi = begin; bi < end; ++bi) {
                const BlockKey key = keys[bi];
                auto &block = *touched_block_ptrs[bi];
                for (int lx = 0; lx < num_voxels_per_block_edge_; ++lx) {
                    const int gx = key.x * num_voxels_per_block_edge_ + lx;
                    for (int ly = 0; ly < num_voxels_per_block_edge_; ++ly) {
                        const int gy = key.y * num_voxels_per_block_edge_ + ly;
                        for (int lz = 0; lz < num_voxels_per_block_edge_; ++lz) {
                            const int gz = key.z * num_voxels_per_block_edge_ + lz;
                            const float wx = (static_cast<float>(gx) + 0.5f) * voxel_edge_m_;
                            const float wy = (static_cast<float>(gy) + 0.5f) * voxel_edge_m_;
                            const float wz = (static_cast<float>(gz) + 0.5f) * voxel_edge_m_;
                            const float cx = w2c(0, 0) * wx + w2c(0, 1) * wy + w2c(0, 2) * wz + w2c(0, 3);
                            const float cy = w2c(1, 0) * wx + w2c(1, 1) * wy + w2c(1, 2) * wz + w2c(1, 3);
                            const float cz = w2c(2, 0) * wx + w2c(2, 1) * wy + w2c(2, 2) * wz + w2c(2, 3);
                            if (cz <= 0.0f) {
                                continue;
                            }

                            // Add 0.5 so the following int cast samples the nearest pixel instead of flooring down/left.
                            const float u_f = cx * intr.fx / cz + intr.cx + 0.5f;
                            const float v_f = cy * intr.fy / cz + intr.cy + 0.5f;
                            if (!(u_f >= 0.0001f && u_f < safe_width && v_f >= 0.0001f && v_f < safe_height)) {
                                continue;
                            }

                            const int u = static_cast<int>(u_f);
                            const int v = static_cast<int>(v_f);
                            const float d = depth_ptr[v * width + u];
                            if (!std::isfinite(d) || d <= 0.0f || d > depth_trunc_f) {
                                continue;
                            }

                            const float du = (static_cast<float>(u) - intr.cx) / intr.fx;
                            const float dv = (static_cast<float>(v) - intr.cy) / intr.fy;
                            const float distance_multiplier = std::sqrt(du * du + dv * dv + 1.0f);
                            const float sdf = (d - cz) * distance_multiplier;
                            if (sdf <= -sdf_trunc_m_) {
                                continue;
                            }

                            const int idx = index_of(lx, ly, lz);
                            Voxel &voxel = block[idx];
                            const float weight_new = voxel.weight + 1.0f;
                            const float tsdf = std::min(1.0f, sdf * sdf_trunc_m_inv);
                            voxel.tsdf = (voxel.tsdf * voxel.weight + tsdf) / weight_new;

                            float r;
                            float g;
                            float b;
                            const float color_u = u_f - 0.5f;
                            const float color_v = v_f - 0.5f;
                            const int u0 = static_cast<int>(std::floor(color_u));
                            const int v0 = static_cast<int>(std::floor(color_v));
                            if (u0 >= 0 && u0 + 1 < width && v0 >= 0 && v0 + 1 < height) {
                                const float fu = color_u - static_cast<float>(u0);
                                const float fv = color_v - static_cast<float>(v0);
                                const float w00 = (1.0f - fu) * (1.0f - fv);
                                const float w10 = fu * (1.0f - fv);
                                const float w01 = (1.0f - fu) * fv;
                                const float w11 = fu * fv;
                                const uint8_t *c00 = color_ptr + (v0 * width + u0) * 3;
                                const uint8_t *c10 = color_ptr + (v0 * width + u0 + 1) * 3;
                                const uint8_t *c01 = color_ptr + ((v0 + 1) * width + u0) * 3;
                                const uint8_t *c11 = color_ptr + ((v0 + 1) * width + u0 + 1) * 3;
                                r = w00 * c00[0] + w10 * c10[0] + w01 * c01[0] + w11 * c11[0];
                                g = w00 * c00[1] + w10 * c10[1] + w01 * c01[1] + w11 * c11[1];
                                b = w00 * c00[2] + w10 * c10[2] + w01 * c01[2] + w11 * c11[2];
                            } else {
                                const uint8_t *rgb = color_ptr + (v * width + u) * 3;
                                r = static_cast<float>(rgb[0]);
                                g = static_cast<float>(rgb[1]);
                                b = static_cast<float>(rgb[2]);
                            }
                            voxel.r = (voxel.r * voxel.weight + r) / weight_new;
                            voxel.g = (voxel.g * voxel.weight + g) / weight_new;
                            voxel.b = (voxel.b * voxel.weight + b) / weight_new;
                            voxel.weight = weight_new;
                        }
                    }
                }
            }
        });
    }

    std::tuple<torch::Tensor, torch::Tensor, int64_t> write_point_cloud(
            const std::string &path,
            int64_t max_points,
            bool select_representatives) const {
        const std::vector<SurfaceTriangle> triangles = surface_triangles();
        const int64_t out_count = surface_sample_count(triangles, max_points);
        auto float_options = torch::dtype(torch::kFloat32).device(torch::kCPU);
        if (out_count == 0) {
            return {torch::empty({0, 3}, float_options), torch::empty({0, 3}, float_options), 0};
        }

        std::ofstream ply_file(path, std::ios::binary);
        TORCH_CHECK(ply_file.good(), "Could not open PLY for writing: ", path);
        write_ply_header(ply_file, out_count);

        std::vector<PlyVertex> chunk;
        chunk.reserve(std::min<size_t>(kPlyChunkVertices, static_cast<size_t>(out_count)));
        std::unordered_map<BlockKey, Representative, BlockKeyHash> representatives;
        auto flush_chunk = [&]() {
            ply_file.write(
                    reinterpret_cast<const char *>(chunk.data()),
                    static_cast<std::streamsize>(chunk.size() * sizeof(PlyVertex)));
            chunk.clear();
        };

        for_each_surface_sample(triangles, max_points, [&](int64_t, const SurfaceVertex &vertex) {
            const PlyVertex ply_vertex = make_ply_vertex(vertex);
            chunk.push_back(ply_vertex);
            if (chunk.size() == kPlyChunkVertices) {
                flush_chunk();
            }

            if (!select_representatives) {
                return;
            }
            const BlockKey cell = {
                    static_cast<int>(std::floor(ply_vertex.x / voxel_edge_m_)),
                    static_cast<int>(std::floor(ply_vertex.y / voxel_edge_m_)),
                    static_cast<int>(std::floor(ply_vertex.z / voxel_edge_m_)),
            };
            const float center_x = (static_cast<float>(cell.x) + 0.5f) * voxel_edge_m_;
            const float center_y = (static_cast<float>(cell.y) + 0.5f) * voxel_edge_m_;
            const float center_z = (static_cast<float>(cell.z) + 0.5f) * voxel_edge_m_;
            const float dx = ply_vertex.x - center_x;
            const float dy = ply_vertex.y - center_y;
            const float dz = ply_vertex.z - center_z;
            float distance_sq = dx * dx;
            distance_sq += dy * dy;
            distance_sq += dz * dz;

            auto inserted = representatives.emplace(cell, Representative{ply_vertex, distance_sq});
            if (!inserted.second && distance_sq < inserted.first->second.distance_sq) {
                inserted.first->second = {ply_vertex, distance_sq};
            }
        });
        if (!chunk.empty()) {
            flush_chunk();
        }
        TORCH_CHECK(ply_file.good(), "Failed while writing PLY: ", path);

        if (!select_representatives) {
            return {torch::empty({0, 3}, float_options), torch::empty({0, 3}, float_options), out_count};
        }

        std::vector<const std::pair<const BlockKey, Representative> *> ordered;
        ordered.reserve(representatives.size());
        for (const auto &item : representatives) {
            ordered.push_back(&item);
        }
        std::sort(ordered.begin(), ordered.end(), [](const auto *a, const auto *b) {
            if (a->first.x != b->first.x) return a->first.x < b->first.x;
            if (a->first.y != b->first.y) return a->first.y < b->first.y;
            return a->first.z < b->first.z;
        });

        const int64_t representative_count = static_cast<int64_t>(ordered.size());
        auto points = torch::empty({representative_count, 3}, float_options);
        auto normals = torch::empty({representative_count, 3}, float_options);
        float *points_ptr = points.data_ptr<float>();
        float *normals_ptr = normals.data_ptr<float>();
        for (int64_t i = 0; i < representative_count; ++i) {
            const PlyVertex &vertex = ordered[static_cast<size_t>(i)]->second.vertex;
            points_ptr[i * 3 + 0] = vertex.x;
            points_ptr[i * 3 + 1] = vertex.y;
            points_ptr[i * 3 + 2] = vertex.z;
            normals_ptr[i * 3 + 0] = vertex.nx;
            normals_ptr[i * 3 + 1] = vertex.ny;
            normals_ptr[i * 3 + 2] = vertex.nz;
        }
        return {points, normals, out_count};
    }

   private:
    std::vector<SurfaceTriangle> surface_triangles() const {
        static const int shifts[8][3] = {
                {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
        };
        static const int tetrahedra[6][4] = {
                {0, 5, 1, 6}, {0, 1, 2, 6}, {0, 2, 3, 6},
                {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6},
        };
        std::vector<BlockKey> keys;
        keys.reserve(all_blocks_.size());
        for (const auto &item : all_blocks_) keys.push_back(item.first);
        std::sort(keys.begin(), keys.end(), [](const BlockKey &a, const BlockKey &b) {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.z < b.z;
        });

        std::vector<SurfaceTriangle> triangles;
        for (const auto &key : keys) {
            const auto &block = all_blocks_.at(key);
            for (int lx = 0; lx < num_voxels_per_block_edge_; ++lx) {
                const int gx = key.x * num_voxels_per_block_edge_ + lx;
                for (int ly = 0; ly < num_voxels_per_block_edge_; ++ly) {
                    const int gy = key.y * num_voxels_per_block_edge_ + ly;
                    for (int lz = 0; lz < num_voxels_per_block_edge_; ++lz) {
                        const int gz = key.z * num_voxels_per_block_edge_ + lz;
                        std::array<const Voxel *, 8> voxels{};
                        bool all_valid = true;
                        bool has_negative = false;
                        bool has_positive = false;
                        for (int i = 0; i < 8; ++i) {
                            voxels[i] = voxel_at(gx + shifts[i][0], gy + shifts[i][1], gz + shifts[i][2]);
                            if (voxels[i] == nullptr || voxels[i]->weight == 0.0f) {
                                all_valid = false;
                                break;
                            }
                            has_negative = has_negative || voxels[i]->tsdf < 0.0f;
                            has_positive = has_positive || voxels[i]->tsdf >= 0.0f;
                        }
                        if (!all_valid || !has_negative || !has_positive) {
                            continue;
                        }

                        std::array<SurfaceVertex, 8> vertices;
                        for (int i = 0; i < 8; ++i) {
                            const int vx = gx + shifts[i][0];
                            const int vy = gy + shifts[i][1];
                            const int vz = gz + shifts[i][2];
                            const Vec3f normal = tsdf_gradient(vx, vy, vz);
                            vertices[i] = {
                                    (static_cast<float>(vx) + 0.5f) * voxel_edge_m_,
                                    (static_cast<float>(vy) + 0.5f) * voxel_edge_m_,
                                    (static_cast<float>(vz) + 0.5f) * voxel_edge_m_,
                                    voxels[i]->r,
                                    voxels[i]->g,
                                    voxels[i]->b,
                                    normal.x,
                                    normal.y,
                                    normal.z,
                            };
                        }
                        for (const auto &tet : tetrahedra) {
                            emit_tetra(voxels, vertices, tet, triangles);
                        }
                    }
                }
            }
        }

        return triangles;
    }

    static int64_t surface_sample_count(const std::vector<SurfaceTriangle> &triangles, int64_t max_points) {
        return max_points > 0 && !triangles.empty() ? max_points : static_cast<int64_t>(triangles.size()) * 3;
    }

    template <typename Emit>
    static void for_each_surface_sample(
            const std::vector<SurfaceTriangle> &triangles,
            int64_t max_points,
            Emit emit) {
        static constexpr double kBarycentricSampleStep1 = 0.7548776662466927;
        static constexpr double kBarycentricSampleStep2 = 0.5698402909980532;
        double total_area = 0.0;
        for (const auto &tri : triangles) {
            total_area += tri.area;
        }
        const int64_t tri_count = static_cast<int64_t>(triangles.size());
        if (max_points <= 0) {
            for (int64_t i = 0; i < tri_count; ++i) {
                emit(i * 3 + 0, triangles[static_cast<size_t>(i)].a);
                emit(i * 3 + 1, triangles[static_cast<size_t>(i)].b);
                emit(i * 3 + 2, triangles[static_cast<size_t>(i)].c);
            }
        } else if (tri_count > 0 && total_area > 0.0) {
            const int64_t out_count = max_points;
            int64_t tri_idx = 0;
            double cumulative = triangles[0].area;
            for (int64_t i = 0; i < out_count; ++i) {
                const double target = (static_cast<double>(i) + 0.5) * total_area / static_cast<double>(out_count);
                while (tri_idx + 1 < tri_count && cumulative < target) {
                    ++tri_idx;
                    cumulative += triangles[tri_idx].area;
                }
                const double r1 = frac((static_cast<double>(i) + 1.0) * kBarycentricSampleStep1);
                const double r2 = frac((static_cast<double>(i) + 1.0) * kBarycentricSampleStep2);
                const float sr1 = static_cast<float>(std::sqrt(r1));
                const float w0 = 1.0f - sr1;
                const float w1 = sr1 * (1.0f - static_cast<float>(r2));
                const float w2 = sr1 * static_cast<float>(r2);
                emit(i, mix_triangle(triangles[static_cast<size_t>(tri_idx)], w0, w1, w2));
            }
        }
    }
    int index_of(int x, int y, int z) const {
        return x * num_voxels_per_block_edge_ * num_voxels_per_block_edge_ + y * num_voxels_per_block_edge_ + z;
    }

    const Voxel *voxel_at(int gx, int gy, int gz) const {
        const int bx = floor_div(gx, num_voxels_per_block_edge_);
        const int by = floor_div(gy, num_voxels_per_block_edge_);
        const int bz = floor_div(gz, num_voxels_per_block_edge_);
        const auto it = all_blocks_.find({bx, by, bz});
        if (it == all_blocks_.end()) {
            return nullptr;
        }
        const int lx = floor_mod(gx, num_voxels_per_block_edge_);
        const int ly = floor_mod(gy, num_voxels_per_block_edge_);
        const int lz = floor_mod(gz, num_voxels_per_block_edge_);
        return &it->second[index_of(lx, ly, lz)];
    }

    float tsdf_derivative(int gx, int gy, int gz, int ax, int ay, int az, const Voxel *center) const {
        const Voxel *neg = voxel_at(gx - ax, gy - ay, gz - az);
        const Voxel *pos = voxel_at(gx + ax, gy + ay, gz + az);
        const bool has_neg = neg != nullptr && neg->weight > 0.0f;
        const bool has_pos = pos != nullptr && pos->weight > 0.0f;
        if (has_neg && has_pos) {
            return (pos->tsdf - neg->tsdf) / (2.0f * voxel_edge_m_);
        }
        if (has_pos) {
            return (pos->tsdf - center->tsdf) / voxel_edge_m_;
        }
        if (has_neg) {
            return (center->tsdf - neg->tsdf) / voxel_edge_m_;
        }
        return 0.0f;
    }

    Vec3f tsdf_gradient(int gx, int gy, int gz) const {
        const Voxel *center = voxel_at(gx, gy, gz);
        if (center == nullptr || center->weight == 0.0f) {
            return {0.0f, 0.0f, 0.0f};
        }
        return {
                tsdf_derivative(gx, gy, gz, 1, 0, 0, center),
                tsdf_derivative(gx, gy, gz, 0, 1, 0, center),
                tsdf_derivative(gx, gy, gz, 0, 0, 1, center),
        };
    }

    static uint8_t clamp_color(float value) {
        value = std::min(255.0f, std::max(0.0f, value));
        return static_cast<uint8_t>(std::round(value));
    }

    static float normal_length_sq(const SurfaceVertex &vertex) {
        return vertex.nx * vertex.nx + vertex.ny * vertex.ny + vertex.nz * vertex.nz;
    }

    static bool has_normal(const SurfaceVertex &vertex) {
        return std::isfinite(vertex.nx) && std::isfinite(vertex.ny) && std::isfinite(vertex.nz) &&
               normal_length_sq(vertex) > 1e-12f;
    }

    static Vec3f normalize(float x, float y, float z) {
        const float len_sq = x * x + y * y + z * z;
        if (!std::isfinite(len_sq) || len_sq <= 1e-12f) {
            return {0.0f, 0.0f, 0.0f};
        }
        const float inv_len = 1.0f / std::sqrt(len_sq);
        return {x * inv_len, y * inv_len, z * inv_len};
    }

    static Vec3f face_normal(const SurfaceVertex &a, const SurfaceVertex &b, const SurfaceVertex &c) {
        const float ux = b.x - a.x;
        const float uy = b.y - a.y;
        const float uz = b.z - a.z;
        const float vx = c.x - a.x;
        const float vy = c.y - a.y;
        const float vz = c.z - a.z;
        return normalize(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    }

    static SurfaceVertex with_fallback_normal(SurfaceVertex vertex, const Vec3f &fallback) {
        if (!has_normal(vertex)) {
            vertex.nx = fallback.x;
            vertex.ny = fallback.y;
            vertex.nz = fallback.z;
        }
        return vertex;
    }

    static SurfaceVertex interpolate(const Voxel *v0, const Voxel *v1, const SurfaceVertex &p0, const SurfaceVertex &p1) {
        const float a0 = std::abs(v0->tsdf);
        const float a1 = std::abs(v1->tsdf);
        const float denom = std::max(a0 + a1, 1e-12f);
        const float t = a0 / denom;
        const Vec3f normal = normalize(
                p0.nx + t * (p1.nx - p0.nx),
                p0.ny + t * (p1.ny - p0.ny),
                p0.nz + t * (p1.nz - p0.nz));
        return {
                p0.x + t * (p1.x - p0.x),
                p0.y + t * (p1.y - p0.y),
                p0.z + t * (p1.z - p0.z),
                (a1 * p0.r + a0 * p1.r) / denom,
                (a1 * p0.g + a0 * p1.g) / denom,
                (a1 * p0.b + a0 * p1.b) / denom,
                normal.x,
                normal.y,
                normal.z,
        };
    }

    static double triangle_area(const SurfaceVertex &a, const SurfaceVertex &b, const SurfaceVertex &c) {
        const double ux = static_cast<double>(b.x) - a.x;
        const double uy = static_cast<double>(b.y) - a.y;
        const double uz = static_cast<double>(b.z) - a.z;
        const double vx = static_cast<double>(c.x) - a.x;
        const double vy = static_cast<double>(c.y) - a.y;
        const double vz = static_cast<double>(c.z) - a.z;
        const double cx = uy * vz - uz * vy;
        const double cy = uz * vx - ux * vz;
        const double cz = ux * vy - uy * vx;
        return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    }

    static void add_triangle(
            std::vector<SurfaceTriangle> &triangles,
            const SurfaceVertex &a,
            const SurfaceVertex &b,
            const SurfaceVertex &c) {
        const double area = triangle_area(a, b, c);
        if (area > 1e-16) {
            const Vec3f fallback = face_normal(a, b, c);
            triangles.push_back({
                    with_fallback_normal(a, fallback),
                    with_fallback_normal(b, fallback),
                    with_fallback_normal(c, fallback),
                    area,
            });
        }
    }

    static void emit_tetra(const std::array<const Voxel *, 8> &voxels,
                           const std::array<SurfaceVertex, 8> &vertices,
                           const int tet[4],
                           std::vector<SurfaceTriangle> &triangles) {
        int inside[4];
        int outside[4];
        int n_inside = 0;
        int n_outside = 0;
        for (int i = 0; i < 4; ++i) {
            const int idx = tet[i];
            if (voxels[idx]->tsdf < 0.0f) {
                inside[n_inside++] = idx;
            } else {
                outside[n_outside++] = idx;
            }
        }
        if (n_inside == 0 || n_inside == 4) {
            return;
        }
        if (n_inside == 1 || n_inside == 3) {
            const bool invert = n_inside == 3;
            const int src = invert ? outside[0] : inside[0];
            const int dst0 = invert ? inside[0] : outside[0];
            const int dst1 = invert ? inside[1] : outside[1];
            const int dst2 = invert ? inside[2] : outside[2];
            SurfaceVertex p0 = interpolate(voxels[src], voxels[dst0], vertices[src], vertices[dst0]);
            SurfaceVertex p1 = interpolate(voxels[src], voxels[dst1], vertices[src], vertices[dst1]);
            SurfaceVertex p2 = interpolate(voxels[src], voxels[dst2], vertices[src], vertices[dst2]);
            add_triangle(triangles, p0, invert ? p2 : p1, invert ? p1 : p2);
            return;
        }

        const int a = inside[0];
        const int b = inside[1];
        const int c = outside[0];
        const int d = outside[1];
        SurfaceVertex ac = interpolate(voxels[a], voxels[c], vertices[a], vertices[c]);
        SurfaceVertex ad = interpolate(voxels[a], voxels[d], vertices[a], vertices[d]);
        SurfaceVertex bc = interpolate(voxels[b], voxels[c], vertices[b], vertices[c]);
        SurfaceVertex bd = interpolate(voxels[b], voxels[d], vertices[b], vertices[d]);
        add_triangle(triangles, ac, bc, ad);
        add_triangle(triangles, bc, bd, ad);
    }

    static SurfaceVertex mix_triangle(const SurfaceTriangle &tri, float w0, float w1, float w2) {
        const Vec3f normal = normalize(
                w0 * tri.a.nx + w1 * tri.b.nx + w2 * tri.c.nx,
                w0 * tri.a.ny + w1 * tri.b.ny + w2 * tri.c.ny,
                w0 * tri.a.nz + w1 * tri.b.nz + w2 * tri.c.nz);
        return {
                w0 * tri.a.x + w1 * tri.b.x + w2 * tri.c.x,
                w0 * tri.a.y + w1 * tri.b.y + w2 * tri.c.y,
                w0 * tri.a.z + w1 * tri.b.z + w2 * tri.c.z,
                w0 * tri.a.r + w1 * tri.b.r + w2 * tri.c.r,
                w0 * tri.a.g + w1 * tri.b.g + w2 * tri.c.g,
                w0 * tri.a.b + w1 * tri.b.b + w2 * tri.c.b,
                normal.x,
                normal.y,
                normal.z,
        };
    }

    static PlyVertex make_ply_vertex(const SurfaceVertex &vertex) {
        const Vec3f normal = normalize(vertex.nx, vertex.ny, vertex.nz);
        return {
                vertex.x,
                vertex.y,
                vertex.z,
                normal.x,
                normal.y,
                normal.z,
                clamp_color(vertex.r),
                clamp_color(vertex.g),
                clamp_color(vertex.b),
                clamp_ply_color((normal.x * 0.5f + 0.5f) * 255.0f),
                clamp_ply_color((normal.y * 0.5f + 0.5f) * 255.0f),
                clamp_ply_color((normal.z * 0.5f + 0.5f) * 255.0f),
        };
    }

    static double frac(double value) {
        return value - std::floor(value);
    }

    float voxel_edge_m_;
    float sdf_trunc_m_;
    float block_edge_m_;
    int num_voxels_per_block_edge_;
    int depth_sampling_stride_;
    int num_voxels_per_block_;
    std::unordered_map<BlockKey, std::vector<Voxel>, BlockKeyHash> all_blocks_;
};

}  // namespace tsdf_ext

void pybind_tsdf_ext(py::module &m) {
    py::class_<tsdf_ext::TSDFVolume>(m, "TSDFVolume")
            .def(py::init<double, double, int, int>(), py::arg("voxel_edge_m"), py::arg("sdf_trunc_m"),
                 py::arg("num_voxels_per_block_edge"), py::arg("depth_sampling_stride"))
            .def("integrate", &tsdf_ext::TSDFVolume::integrate, py::arg("depth"), py::arg("color"),
                 py::arg("intrinsics"), py::arg("extrinsic"), py::arg("depth_trunc"),
                 py::call_guard<py::gil_scoped_release>())
            .def("write_point_cloud", &tsdf_ext::TSDFVolume::write_point_cloud, py::arg("path"),
                 py::arg("max_points"), py::arg("select_representatives"),
                 py::call_guard<py::gil_scoped_release>());
}
