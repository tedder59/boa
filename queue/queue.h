#ifndef BOA_QUEUE_QUEUE_H_
#define BOA_QUEUE_QUEUE_H_

#include "array/array.h"
#include "linked_list/linked_list.h"

/**
 * 数组队列， 入队，出队
 */
template <typename T>
class ArrayQueue
{
public:
    ArrayQueue(int queue_num)
        : m_queue(queue_num)
        , head(0)
        , tail(0) { }

    void push_back(const T& v) {
        assert(!full());
        m_queue[tail] = v;
        tail = (tail + 1) % m_queue.size();
    }

    T pop_front() {
        assert(!empty());
        size_t curr = head;
        head = (head + 1) % m_queue.size();
        return m_queue[curr];
    }

    bool full() {
        return (tail + 1) % m_queue.size() == head;
    }

    bool empty() {
        return head == tail;
    }

private:
    Array<T>    m_queue;
    size_t      head;
    size_t      tail;
};

/**
 * 链表队列，动态容量，入队，出队
 */
template <typename T>
class LinkQueue : public SingleLinkedList<T>
{
public:
    void push_back(const T& t) {
        SingleLinkedList<T>::insert_tail(t);
    }

    T pop_front() {
        assert(SingleLinkedList<T>::head);
        SingleLinkedNode<T>* curr = SingleLinkedList<T>::head;
        SingleLinkedList<T>::head = curr->next;
        T rsl = curr->val;
        delete curr;
        
        return rsl;
    }
};

#endif