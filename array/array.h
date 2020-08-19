#ifndef BOA_ARRAY_ARRAY_H_
#define BOA_ARRAY_ARRAY_H_

#include "common/utils.h"
#include <functional>
#include <assert.h>

/*
 * 数组类型,不支持动态扩容，能够设置数组的大小和通过下标来访问元素
 */
template <typename T>
class Array
{
public:
    Array(size_t size)
        : m_nSize(size)
        , m_nCapacity(size) {
        m_pElem = new T[size];
    }

    ~Array() {
        if (m_pElem) delete[] m_pElem;
    }

    inline void resize(size_t new_size) {
        if (new_size <= m_nCapacity)
        {
            m_nSize = new_size;
            return;
        }
        
        T* new_arr = new T[new_size];
        for (size_t i = 0; i < m_nSize; ++i)
        {
            new_arr[i] = m_pElem[i];
        }

        delete[] m_pElem;
        m_pElem = new_arr;
        m_nCapacity = m_nSize = new_size;
    }

    inline T& operator[](size_t idx) {
        assert(idx < m_nSize);
        return m_pElem[idx];
    }

    inline const T& operator[](size_t idx) const {
        assert(idx < m_nSize);
        return m_pElem[idx];
    }

    inline size_t size() const {
        return m_nSize;
    }

protected:
    T*      m_pElem;
    size_t  m_nSize;
    size_t  m_nCapacity;
};

/*
 * 有序数组(升序），支持动态扩容，能够根据下标访问元素，添加和删除元素
 * 支持二分查找
 */
template <typename T>
class OrderedArray 
{
public:
    OrderedArray(size_t size)
        : m_nSize(0)
        , m_nCapacity(size) {
        m_pElem = new T[size];
    }

    ~OrderedArray() {
        if (m_pElem) delete[] m_pElem;
    }

    inline const T& operator[](size_t idx) const {
        assert(idx < m_nSize);
        return m_pElem[idx];
    }

    inline T& operator[](size_t idx) {
        assert(idx < m_nSize);
        return m_pElem[idx];
    }

    inline size_t size() const {
        return m_nSize;
    }

    void insert(const T& elem) {
        if ((m_nSize + 1) <= m_nCapacity)
        {
            resize(_increase_rate * m_nCapacity);
        }

        size_t idx = 0;
        bool ret = least_large(elem, idx);
        if (!ret)
        {
            m_pElem[m_nSize++] = elem;
            return;
        }

        for (size_t rear = m_nSize++; rear > idx; --rear)
        {
            m_pElem[rear] = m_pElem[rear - 1];
        }
       
        m_pElem[idx] = elem;
    }

    void remove(const T& elem) {
        if (m_nSize <= 0) return;
        
        size_t start = 0;
        bool ret = least_equal(elem, start);
        if (!ret) return;

        size_t end = 0;
        most_equal(elem, end);
        
        size_t offset = end - start + 1;
        m_nSize -= offset;
        while (start < m_nSize)
        {
            m_pElem[start] = m_pElem[start + offset];
            ++start;
        }
    }

    bool least_equal(const T& elem, size_t& idx) {
        if (m_nSize <= 0) return false;
        long l = 0;
        long r = m_nSize - 1;

        while (l <= r)
        {
            long m = l + ((r - l) >> 1);
            if (m_pElem[m] >= elem)
            {
                r = m - 1;
            }
            else
            {
                l = m + 1;
            }
        }
        
        if (l >= 0 && m_pElem[l] == elem)
        {
            idx = l;
            return true;
        }

        return false;
    }

    bool most_equal(const T& elem, size_t& idx) {
        if (m_nSize <= 0) return false;
        long l = 0;
        long r = m_nSize - 1;

        while (l <= r)
        {
            long m = l + ((r - l) >> 1);
            if (m_pElem[m] <= elem)
            {
                l = m + 1;
            }
            else
            {
                r = m - 1;
            }
            
        }
        
        if (r >= 0 && m_pElem[r] == elem)
        {
            idx = r;
            return true;
        }
        
        return false;
    }

    bool least_large(const T& elem, size_t& idx) {
        if (m_nSize <= 0) return false;
        long l = 0;
        long r = m_nSize - 1;
        
        while (l <= r)
        {
            long m = l + ((r - l) >> 1);
            if (m_pElem[m] <= elem)
            {
                l = m + 1;
            }
            else
            {
                r = m - 1;
            }
        }
        
        if (l < m_nSize)
        {
            idx = l;
            return true;
        }

        return false;
    }

    bool most_less(const T& elem, size_t& idx) {
        if (m_nSize <= 0) return false;
        long l = 0;
        long r = m_nSize - 1;

        while (l <= r)
        {
            long m = l + ((r - l) >> 1);
            if (m_pElem[m] >= elem)
            {
                r = m - 1;
            }
            else
            {
                l = m + 1;
            }
        }
        
        if (r >= 0)
        {
            idx = r;
            return true;
        }

        return false;
    }

private:
    inline void resize(size_t size) {
        if (m_nCapacity > size) return;
        
        T* new_arr = new T[size];
        assert(new_arr);

        for (size_t i = 0; i < m_nSize; ++i)
        {
            new_arr[i] = m_pElem[i];
        }

        delete[] m_pElem;
        m_pElem = new_arr;
        m_nCapacity = size;
    }

private:
    friend void MergeApply<OrderedArray<T>>::merge(
        const OrderedArray<T>& l0,
        const OrderedArray<T>& l1,
        OrderedArray<T>& out);

    T*      m_pElem;
    size_t  m_nSize;
    size_t  m_nCapacity;

    const float _increase_rate {1.2f};
};

/*
 * 合并有序数组
 */
template <>
template <typename T>
class MergeApply<OrderedArray<T>>
{
public:
    static void merge(const OrderedArray<T>& l0,
                      const OrderedArray<T>& l1,
                      OrderedArray<T>& out) {
        size_t new_size = l0.size() + l1.size();
        out.resize(new_size);
        out.m_nSize = new_size;

        if (new_size <= 0 ) return;

        size_t it0 = 0;
        size_t it1 = 0;
        size_t it = 0;

        while (it < new_size)
        {
            if (it0 < l0.size() && it1 < l1.size())
            {
                if (l0[it0] < l1[it1])
                {
                    out[it++] = l0[it0++];
                }
                else
                {
                    out[it++] = l1[it1++];
                }
            }
            else if (it0 < l0.size())
            {
                while (it0 < l0.size())
                {
                    out[it++] = l0[it0++];
                }
            }
            else 
            {
                while (it1 < l1.size())
                {
                    out[it++] = l1[it1++];
                }
            }
        }
    }
};

#endif // BOA_ARRAY_ARRAY_H_