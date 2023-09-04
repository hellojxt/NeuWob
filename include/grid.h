#include "common.h"

NWOB_NAMESPACE_BEGIN

template <typename T, int N>
class CellList
{
    public:
        int num;
        T data[N];
        inline __device__ void atomic_append(T value)
        {
            if (num < N)
            {
                int index = atomicAdd(&num, 1);
                data[index] = value;
            }
        }
        HOST_DEVICE inline void append(T value)
        {
            if (num < N)
            {
                data[num] = value;
                num++;
            }
        }
        HOST_DEVICE inline T operator[](int index) const { return data[index]; }
        HOST_DEVICE inline int size() { return num; }
};

class Grid
{
    public:
        float cell_length;
        int3 size;
        float3 min_pos, max_pos;
        inline HOST_DEVICE void check_safe(float3 pos) const noexcept
        {
            if (pos.x < min_pos.x || pos.x > max_pos.x || pos.y < min_pos.y || pos.y > max_pos.y || pos.z < min_pos.z ||
                pos.z > max_pos.z)
            {
                printf("pos: %f %f %f\n", pos.x, pos.y, pos.z);
                printf("min_pos: %f %f %f\n", min_pos.x, min_pos.y, min_pos.z);
                printf("max_pos: %f %f %f\n", max_pos.x, max_pos.y, max_pos.z);
            }
        }

        inline HOST_DEVICE void check_safe(int3 index) const noexcept
        {
            if (index.x < 0 || index.x >= size.x || index.y < 0 || index.y >= size.y || index.z < 0 ||
                index.z >= size.z)
            {
                printf("index: %d %d %d\n", index.x, index.y, index.z);
                printf("size: %d %d %d\n", size.x, size.y, size.z);
            }
        }

        inline HOST_DEVICE void check_safe(int flat_index) const noexcept
        {
            if (flat_index < 0 || flat_index >= size.x * size.y * size.z)
            {
                printf("flat_index: %d\n", flat_index);
                printf("size: %d %d %d\n", size.x, size.y, size.z);
            }
        }

        inline HOST_DEVICE int get_cell_num() const noexcept { return size.x * size.y * size.z; }

        inline HOST_DEVICE int3 get_cell_index(float3 pos) const noexcept
        {
            auto cell_index = make_int3((pos.x - min_pos.x) / cell_length, (pos.y - min_pos.y) / cell_length,
                                        (pos.z - min_pos.z) / cell_length);
#ifdef MEM_CHECK
            check_safe(pos);
            check_safe(cell_index);
#endif
            return cell_index;
        }

        inline HOST_DEVICE int3 get_cell_index(int flat_index) const noexcept
        {
            int3 index;
            index.z = flat_index / (size.x * size.y);
            flat_index -= index.z * size.x * size.y;
            index.y = flat_index / size.x;
            index.x = flat_index % size.x;
#ifdef MEM_CHECK
            check_safe(flat_index);
            check_safe(index);
#endif
            return index;
        }

        inline HOST_DEVICE int get_flat_index(int3 index) const noexcept
        {
#ifdef MEM_CHECK
            check_safe(index);
#endif
            return index.x + index.y * size.x + index.z * size.x * size.y;
        }

        inline HOST_DEVICE int get_flat_index(float3 pos) const noexcept { return get_flat_index(get_cell_index(pos)); }

        inline HOST_DEVICE float3 get_cell_center(int3 index) const noexcept
        {
#ifdef MEM_CHECK
            check_safe(index);
#endif
            return make_float3(min_pos.x + (index.x + 0.5f) * cell_length, min_pos.y + (index.y + 0.5f) * cell_length,
                               min_pos.z + (index.z + 0.5f) * cell_length);
        }

        inline HOST_DEVICE float3 get_cell_center(int flat_index) const noexcept
        {
            return get_cell_center(get_cell_index(flat_index));
        }
};

NWOB_NAMESPACE_END