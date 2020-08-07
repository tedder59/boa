#ifndef BOA_ARRAY_ARRAY_H_
#define BOA_ARRAY_ARRAY_H_

#include <functional>
#include <exception>
#include <iostream>
#include <assert.h>

template <typename T>
class Array
{
public:
    Array(size_t size)
        : m_nCapacity(size)
        , m_nSize(0)
    {
        m_pElem = new T[size];
    }

    ~Array() {
        if (m_pElem) delete[] m_pElem;
    }

    void resize(size_t new_size) {
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

    T& operator[](size_t idx)
    {
        return m_pElem[idx];
    }

    const T& operator[](size_t idx) const
    {
        return m_pElem[idx];
    }

    void clear() { m_nSize = 0; }

    size_t size() const { 
        return m_nSize;
    }

    size_t capacity() const {
        return m_nCapacity;
    }

private:
    template <typename T1>
    friend class OrderedArray;
    T*      m_pElem;
    size_t  m_nSize;
    size_t  m_nCapacity;
};

template <typename T>
class OrderedArray 
{
public:
    OrderedArray(size_t size) : m_array(size) {}

    T& operator[](size_t idx) {
        return m_array[idx];
    }

    const T& operator[](size_t idx) const {
        return m_array[idx];
    }

    size_t size() const {
        return m_array.size();
    }

    void insert(const T& elem) {
        assert((m_array.m_nSize + 1) <= m_array.m_nCapacity);
        long idx = bsearch_least_large_equal(elem);

        if (idx == -1)
        {
            m_array[m_array.m_nSize++] = elem;
            return;
        }

        if (m_array[idx] == elem) ++idx;

        size_t start = m_array.m_nSize;
        while(start > idx)
        {
            m_array[start] = m_array[start - 1];
            --start;
        }
       
        m_array[idx] = elem;
        ++m_array.m_nSize;
    }

    void erease(const T& elem) {
        if (m_array.m_nSize <= 0) return;
        
        long start = bsearch_least_equal(elem);
        if (start < 0) return;

        long end = bsearch_most_equal(elem);
        if (end < 0) return;
        
        size_t offset = end - start + 1;
        m_array.m_nSize -= offset;

        while (start < m_array.m_nSize)
        {
            m_array[start] = m_array[start + offset];
            ++start;
        }
    }

    void merge(const OrderedArray& other) {
        size_t new_size = m_array.m_nSize + other.size();
        assert(new_size <= m_array.m_nCapacity);
        if (new_size <= 0) return;

        long start = new_size - 1;
        long offset = other.size();
        while(start >= offset)
        {
            m_array[start] = m_array[start - offset];
            --start;
        }
        
        size_t it0 = offset;
        size_t it1 = 0;
        size_t it = 0;

        while (it < new_size)
        {
            if (it0 < new_size && it1 < offset)
            {
                if (m_array[it0] < other[it1])
                {
                    m_array[it++] = m_array[it0++];
                }
                else
                {
                    m_array[it++] = other[it1++];
                }
            }
            else if (it1 < offset)
            {
                while (it1 < offset)
                {
                    m_array[it++] = other[it1++];
                }
            }
            else
            {
                break;
            }
        }

        m_array.m_nSize = new_size;
    }

    long bsearch_least_equal(const T& elem) {
        size_t n = m_array.size();
        if (n <= 0) return -1;

        long left = 0;
        long right = n - 1;

        while (left <= right )
        {
            long middle = left + ((right - left) >> 1);

            if (m_array[middle] >= elem)
            {
                right = middle - 1;
            }
            else
            {
                left = middle + 1;
            }
        }

        if (left >= 0 && m_array[left] == elem)
        {
            return left;
        }
        
        return -1;
    }

    long bsearch_most_equal(const T& elem) {
        size_t n = m_array.size();
        if (n <= 0) return -1;

        long left = 0;
        long right = n - 1;

        while (left <= right )
        {
            long middle = left + ((right - left) >> 1);

            if (m_array[middle] <= elem)
            {
                left = middle + 1;
            }
            else
            {
                right = middle - 1;
            }
        }

        if (right < n && m_array[right] == elem)
        {
            return right;
        }
        
        return -1;
    }

    long bsearch_least_large_equal(const T& elem) {
        size_t n = m_array.size();
        if (n <= 0) return -1;

        long left = 0;
        long right = n - 1;

        while (left <= right )
        {
            long middle = left + ((right - left) >> 1);

            if (m_array[middle] <= elem)
            {
                left = middle + 1;
            }
            else
            {
                right = middle - 1;
            }
        }

        if (left < n)
        {
            return left;
        }
        
        return -1;
    }

    long bsearch_most_less_equal(const T& elem) {
        size_t n = m_array.size();
        if (n <= 0) return -1;

        long left = 0;
        long right = n - 1;

        while (left <= right )
        {
            long middle = left + ((right - left) >> 1);

            if (m_array[middle] >= elem)
            {
                right = middle - 1;
            }
            else
            {
                left = middle + 1;
            }
        }

        if (right >= 0)
        {
            return right;
        }
        
        return -1;
    }

private:
    template <typename T1>
    friend size_t merge(const OrderedArray<T1>& first,
                        const OrderedArray<T1>& second,
                        OrderedArray<T1>& merged);

    Array<T>    m_array;
};

template <typename T>
size_t merge(const OrderedArray<T>& first,
             const OrderedArray<T>& second,
             OrderedArray<T>& merged) {
    size_t new_size = first.size() + second.size();
    merged.m_array.resize(new_size);
    
    if (new_size <= 0 ) return 0;

    size_t it0 = 0;
    size_t it1 = 0;
    size_t it = 0;

    while (it < new_size)
    {
        if (it0 < first.size() && it1 < second.size())
        {
            if (first[it0] < second[it1])
            {
                merged[it++] = first[it0++];
            }
            else
            {
                merged[it++] = second[it1++];
            }
        }
        else if (it0 < first.size())
        {
            while (it0 < first.size())
            {
                merged[it++] = first[it0++];
            }
        }
        else 
        {
            while (it1 < second.size())
            {
                merged[it++] = second[it1++];
            }
        }
    }

    return new_size;
}

#endif // BOA_ARRAY_ARRAY_H_