#ifndef BOA_HASHTABLE_LRU_H_
#define BOA_HASHTABLE_LRU_H_

#include "linked_list/linked_list.h"
#include <unordered_map>
#include <iostream>

template <typename K, typename V>
struct Node
{
    K k_;
    V v_;
};

template <typename T>
class TypeTraits
{
    typedef typename T::key_type key_type;
    typedef typename T::value_type value_type;
};

template <>
template <typename K, typename V>
class TypeTraits<Node<K, V>>
{
public:
    typedef K key_type;
    typedef V value_type;
};

template <typename T>
class LRU
{
public:
    LRU(size_t size) : size_(size), used_(0) {}
    typename TypeTraits<T>::value_type get(typename TypeTraits<T>::key_type key) {
        if (m_hashtable.find(key) == m_hashtable.end())
        {
            DoubleLinkedNode<T>* node = push_front(key);
            std::cout << node->val.v_ << "\tnot in cache" << std::endl;
            
            m_hashtable[key] = node;
            return node->val.v_;
        }
        else
        {
            DoubleLinkedNode<T>* node = m_hashtable[key];
            std::cout << node->val.v_ << "\tin cache" << std::endl;
            move_to_front(node);
            return node->val.v_;
        }
    }

private:
    DoubleLinkedNode<T>* push_front(typename TypeTraits<T>::key_type key) {
        DoubleLinkedNode<T>* node;
        if (used_ >= size_)
        {
            node = m_list.tail;
            m_list.tail = node->prev;
            m_list.tail->next = node->next;

            m_hashtable.erase(node->val.k_);
        }
        else
        {
            node = new DoubleLinkedNode<T>();
            used_++;
        }

        // assemble value
        node->val.k_ = key;
        node->val.v_ = std::to_string(key);
        
        // push front
        if (m_list.head)
        {
            node->prev = m_list.head->prev;
            node->next = m_list.head;
            m_list.head->prev = node;
        }
        else
        {
            node->prev = nullptr;
            node->next = nullptr;
            m_list.tail = node;
        }
        
        m_list.head = node;
        return node;
    }

    void move_to_front(DoubleLinkedNode<T>* node) {
        if (node->prev)
            node->prev->next = node->next;
        if (node->next)
            node->next->prev = node->prev;

        node->next = m_list.head;
        node->prev = m_list.head->prev;
        m_list.head->prev = node;
        m_list.head = node;
    }

private:
    DoubleLinkedList<T> m_list;
    std::unordered_map<typename TypeTraits<T>::key_type, DoubleLinkedNode<T>*> m_hashtable;

    size_t  size_;
    size_t  used_;
};

#endif // BOA_HASHTABLE_LRU_H_