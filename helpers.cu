#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>
#include "cuda_fp16.h"

#define CUDA_DEVICE_INLINE __device__ __forceinline__

typedef struct
{
    int x;
} Coord1D;

typedef struct
{
    int x, y;
} Coord2D;

typedef struct
{
    int x, y, z;
} Coord3D;

typedef struct
{
    int x, y, z, t;
} Coord4D;

typedef struct
{
    int x, y, z, t, u;
} Coord5D;

template <typename T, int Dims, typename Derived>
class BaseTensor
{
public:
    CUDA_DEVICE_INLINE
    BaseTensor(void *ptr)
        : startPtr(reinterpret_cast<T *>(ptr)), endPtr(reinterpret_cast<T *>(ptr) + static_cast<const Derived *>(this)->totalSize()) {}

    T *startPtr;
    T *endPtr;

    CUDA_DEVICE_INLINE T get() const { return startPtr[0]; }
    CUDA_DEVICE_INLINE T get(int x) const { return startPtr[x]; }
    CUDA_DEVICE_INLINE T get(int x, int y) const { return startPtr[x * static_cast<const Derived *>(this)->stride().x + y]; }
    CUDA_DEVICE_INLINE T get(int x, int y, int z) const { return startPtr[x * static_cast<const Derived *>(this)->stride().x + y * static_cast<const Derived *>(this)->stride().y + z]; }

    CUDA_DEVICE_INLINE void set(const T value) { startPtr[0] = value; }
    CUDA_DEVICE_INLINE void set(int x, const T value) { startPtr[x] = value; }
    CUDA_DEVICE_INLINE void set(int x, int y, const T value) { startPtr[x * static_cast<const Derived *>(this)->stride().x + y] = value; }
    CUDA_DEVICE_INLINE void set(int x, int y, int z, const T value) { startPtr[x * static_cast<const Derived *>(this)->stride().x + y * static_cast<const Derived *>(this)->stride().y + z] = value; }

    CUDA_DEVICE_INLINE T *get_ptr() { return startPtr; }
    CUDA_DEVICE_INLINE T *get_ptr(int x) { return &startPtr[x]; }
    CUDA_DEVICE_INLINE T *get_ptr(int x, int y) { return &startPtr[x * static_cast<const Derived *>(this)->stride().x + y]; }
    CUDA_DEVICE_INLINE T *get_ptr(int x, int y, int z) { return &startPtr[x * static_cast<const Derived *>(this)->stride().x + y * static_cast<const Derived *>(this)->stride().y + z]; }

    template <typename U>
    CUDA_DEVICE_INLINE U get_reinterpreted() const
    {
        return *reinterpret_cast<const U *>(startPtr);
    }

    template <typename U>
    CUDA_DEVICE_INLINE U get_reinterpreted(int x) const
    {
        return reinterpret_cast<const U *>(startPtr)[x];
    }

    template <typename U>
    CUDA_DEVICE_INLINE U get_reinterpreted(int x, int y) const
    {
        return reinterpret_cast<const U *>(startPtr)[(x * static_cast<const Derived *>(this)->stride().x + y) * sizeof(T) / sizeof(U)];
    }

    template <typename U>
    CUDA_DEVICE_INLINE U get_reinterpreted(int x, int y, int z) const
    {
        return reinterpret_cast<const U *>(startPtr)[(x * static_cast<const Derived *>(this)->stride().x + y * static_cast<const Derived *>(this)->stride().y + z) * sizeof(T) / sizeof(U)];
    }

    template <typename U>
    CUDA_DEVICE_INLINE void set_reinterpreted(U value)
    {
        *reinterpret_cast<U *>(startPtr) = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE void set_reinterpreted(int x, U value)
    {
        reinterpret_cast<U *>(startPtr)[x] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE void set_reinterpreted(int x, int y, U value)
    {
        reinterpret_cast<U *>(startPtr)[(x * static_cast<const Derived *>(this)->stride().x + y) * sizeof(T) / sizeof(U)] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE void set_reinterpreted(int x, int y, int z, U value)
    {
        reinterpret_cast<U *>(startPtr)[(x * static_cast<const Derived *>(this)->stride().x + y * static_cast<const Derived *>(this)->stride().y + z) * sizeof(T) / sizeof(U)] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE U *get_ptr_reinterpreted()
    {
        return reinterpret_cast<U *>(startPtr);
    }

    template <typename U>
    CUDA_DEVICE_INLINE U *get_ptr_reinterpreted(int x)
    {
        return &reinterpret_cast<U *>(startPtr)[x];
    }

    template <typename U>
    CUDA_DEVICE_INLINE U *get_ptr_reinterpreted(int x, int y)
    {
        return &reinterpret_cast<U *>(startPtr)[(x * static_cast<const Derived *>(this)->stride().x + y) * sizeof(T) / sizeof(U)];
    }

    template <typename U>
    CUDA_DEVICE_INLINE U *get_ptr_reinterpreted(int x, int y, int z)
    {
        return &reinterpret_cast<U *>(startPtr)[(x * static_cast<const Derived *>(this)->stride().x + y * static_cast<const Derived *>(this)->stride().y + z) * sizeof(T) / sizeof(U)];
    }
};

template <typename T>
class SmemTensor0D : public BaseTensor<T, 0, SmemTensor0D<T>>
{
public:
    using BaseTensor<T, 0, SmemTensor0D<T>>::BaseTensor;
    CUDA_DEVICE_INLINE int totalSize() const { return 1; }
};

template <typename T, int ShapeX>
class SmemTensor1D : public BaseTensor<T, 1, SmemTensor1D<T, ShapeX>>
{
public:
    using BaseTensor<T, 1, SmemTensor1D<T, ShapeX>>::BaseTensor;
    CUDA_DEVICE_INLINE Coord1D shape() const { return {ShapeX}; }
    CUDA_DEVICE_INLINE int totalSize() const { return ShapeX; }
};

template <typename T, int ShapeX, int ShapeY>
class SmemTensor2D : public BaseTensor<T, 2, SmemTensor2D<T, ShapeX, ShapeY>>
{
public:
    using BaseTensor<T, 2, SmemTensor2D<T, ShapeX, ShapeY>>::BaseTensor;
    CUDA_DEVICE_INLINE Coord2D shape() const { return {ShapeX, ShapeY}; }
    CUDA_DEVICE_INLINE Coord1D stride() const { return {ShapeY}; }
    CUDA_DEVICE_INLINE int totalSize() const { return ShapeX * ShapeY; }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeY> get_child(int x)
    {
        return SmemTensor1D<T, ShapeY>(this->startPtr + x * this->stride().x);
    }
};

template <typename T, int ShapeX, int ShapeY, int ShapeZ>
class SmemTensor3D : public BaseTensor<T, 3, SmemTensor3D<T, ShapeX, ShapeY, ShapeZ>>
{
public:
    using BaseTensor<T, 3, SmemTensor3D<T, ShapeX, ShapeY, ShapeZ>>::BaseTensor;
    CUDA_DEVICE_INLINE Coord3D shape() const { return {ShapeX, ShapeY, ShapeZ}; }
    CUDA_DEVICE_INLINE Coord2D stride() const { return {ShapeY * ShapeZ, ShapeZ}; }
    CUDA_DEVICE_INLINE int totalSize() const { return ShapeX * ShapeY * ShapeZ; }

    CUDA_DEVICE_INLINE
    SmemTensor2D<T, ShapeY, ShapeZ> get_child(int x)
    {
        return SmemTensor2D<T, ShapeY, ShapeZ>(this->startPtr + x * this->stride().x);
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeZ> get_child(int x, int y)
    {
        return SmemTensor1D<T, ShapeZ>(this->startPtr + x * this->stride().x + y * this->stride().y);
    }
};

template <typename T>
class GMemTensor0D : public BaseTensor<T, 0, GMemTensor0D<T>>
{
public:
    using BaseTensor<T, 0, GMemTensor0D<T>>::BaseTensor;
    CUDA_DEVICE_INLINE int totalSize() const { return 1; }
};

template <typename T>
class GMemTensor1D : public BaseTensor<T, 1, GMemTensor1D<T>>
{
private:
    int shapeX;

public:
    CUDA_DEVICE_INLINE
    GMemTensor1D(void *gmemPtr, int x) : BaseTensor<T, 1, GMemTensor1D<T>>(gmemPtr), shapeX(x) {}

    CUDA_DEVICE_INLINE Coord1D shape() const { return {shapeX}; }
    CUDA_DEVICE_INLINE int totalSize() const { return shapeX; }
};

template <typename T>
class GMemTensor2D : public BaseTensor<T, 2, GMemTensor2D<T>>
{
private:
    int shapeX, shapeY;

public:
    CUDA_DEVICE_INLINE
    GMemTensor2D(void *gmemPtr, int x, int y) : BaseTensor<T, 2, GMemTensor2D<T>>(gmemPtr), shapeX(x), shapeY(y) {}

    CUDA_DEVICE_INLINE Coord2D shape() const { return {shapeX, shapeY}; }
    CUDA_DEVICE_INLINE Coord1D stride() const { return {shapeY}; }
    CUDA_DEVICE_INLINE int totalSize() const { return shapeX * shapeY; }

    CUDA_DEVICE_INLINE
    GMemTensor1D<T> get_child(int x)
    {
        return GMemTensor1D<T>(this->startPtr + x * this->stride().x, shapeY);
    }
};

template <typename T>
class GMemTensor3D : public BaseTensor<T, 3, GMemTensor3D<T>>
{
private:
    int shapeX, shapeY, shapeZ;

public:
    CUDA_DEVICE_INLINE
    GMemTensor3D(void *gmemPtr, int x, int y, int z) : BaseTensor<T, 3, GMemTensor3D<T>>(gmemPtr), shapeX(x), shapeY(y), shapeZ(z) {}

    CUDA_DEVICE_INLINE Coord3D shape() const { return {shapeX, shapeY, shapeZ}; }
    CUDA_DEVICE_INLINE Coord2D stride() const { return {shapeY * shapeZ, shapeZ}; }
    CUDA_DEVICE_INLINE int totalSize() const { return shapeX * shapeY * shapeZ; }

    CUDA_DEVICE_INLINE
    GMemTensor2D<T> get_child(int x)
    {
        return GMemTensor2D<T>(this->startPtr + x * this->stride().x, shapeY, shapeZ);
    }

    CUDA_DEVICE_INLINE
    GMemTensor1D<T> get_child(int x, int y)
    {
        return GMemTensor1D<T>(this->startPtr + x * this->stride().x + y * this->stride().y, shapeZ);
    }
};