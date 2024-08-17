#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        used += size;
        peak = std::max(used, peak);

        for (auto &[offset, len] : free_blocks) {
            if (len >= size) {
                len -= size;
                return offset + len;
            }
        }

        auto offset = total_size;
        total_size += size;
        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        used -= size;

        if (addr + size == total_size) {
            total_size -= size;
            return;
        }
        
        auto it = free_blocks.emplace(addr, size).first;

        // 是否能与后面的内存合并
        if (auto next_it = std::next(it); next_it != std::end(free_blocks)) {
            if (next_it->first == addr + size) {
                size += next_it->second;
                free_blocks.erase(next_it);
            }
        }

        // 是否能与前面的内存合并
        if (it != std::begin(free_blocks)) {
            if (auto prev_it = std::prev(it); prev_it->first + prev_it->second == addr) {
                prev_it->second += size;
                free_blocks.erase(it);
            }
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
