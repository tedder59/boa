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
    Array(size_t size) : m_nCapacity(size), m_nSize(0) {
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

    T& operator[](size_t idx) {
        // assert(idx < m_nSize);
        return m_pElem[idx];
    }

    const T& operator[](size_t idx) const {
        // assert(idx < m_nSize);
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
        long right = m_array.size();
        if (right > 0)
        {
            right = bsearch(0, right - 1, elem);
            right = abs(right);

            if (right < m_array.m_nSize && m_array[right] < elem)
            {
                ++right;
            }
            
            size_t start = m_array.m_nSize;
            while(start > right)
            {
                m_array[start] = m_array[start - 1];
                --start;
            }
        }
       
        m_array[right] = elem;
        ++m_array.m_nSize;
    }

    void erease(const T& elem) {
        if (m_array.m_nSize <= 0) return;

        size_t right = m_array.m_nSize - 1;
        long idx = bsearch(0, right, elem);
        if (idx < 0) return;

        size_t start = idx;
        while (m_array[start - 1] == m_array[start])
        {
            --start;
        }

        size_t end = idx;
        while (m_array[end + 1] == m_array[end])
        {
            ++end;
        }
        
        size_t offset = end - start + 1;
        m_array.m_nSize -= offset;

        while (start < m_array.m_nSize)
        {
            m_array[start] = m_array[start + offset];
            ++start;
        }
    }

    void merge(const OrderedArray& other) {
        size_t new_size = m_array.m_nSize + other.m_array.m_nSize;
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

protected:
    long bsearch(long left, long right, const T& elem) {
        while (left <= right )
        {
            long middle = left + ((right - left) >> 1);

            if (m_array[middle] == elem)
            {
                return middle;
            }
            else if (m_array[middle] < elem)
            {
                left = middle + 1;
            }
            else
            {
                right = middle - 1;
            }
        }
        
        return -std::max(left, right);
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