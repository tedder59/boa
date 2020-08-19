#ifndef BOA_STACK_STACK_H_
#define BOA_STACK_STACK_H_

#include "array/array.h"
#include "linked_list/linked_list.h"

/**
 * 数组栈， 入栈， 出栈
 */
template <typename T>
class ArrayStack
{
public:
    ArrayStack(int stack_num)
        : m_stack(stack_num)
        , size(0) {}

    void push(const T& v) {
        assert(size < m_stack.size());
        m_stack[size++] = v;
    }

    T pop() {
        assert(size > 0);
        --size;
        return m_stack[size];
    }

    T peek() {
        assert(size > 0);
        return m_stack[size - 1];
    }

private:
    Array<T>    m_stack;
    size_t      size;
};

/**
 * 链表栈， 入栈，出栈
 */
template <typename T>
class LinkStack : public SingleLinkedList<T>
{
public:
    void push(const T& t) {
        SingleLinkedList<T>::insert_head(t);
    }

    T pop() {
        assert(SingleLinkedList<T>::head);
        SingleLinkedNode<T>* curr = SingleLinkedList<T>::head;
        SingleLinkedList<T>::head = curr->next;
        T rsl = curr->val;
        delete curr;

        return rsl;
    }

    T peek() {
        assert(SingleLinkedList<T>::head);
        return SingleLinkedList<T>::head->val;
    }
};

#endif