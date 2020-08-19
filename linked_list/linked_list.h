#ifndef BOA_LINKED_LIST_LINKED_LIST_H_
#define BOA_LINKED_LIST_LINKED_LIST_H_

#include <functional>
#include <assert.h>

/**
 * 单链表结点
 */
template <typename T>
struct SingleLinkedNode
{
    SingleLinkedNode<T>* next;
    T                    val;
};

/**
 * 单链表,在表头添加元素，在表尾添加元素
 * 删除指定元素，链表反转，求中间结点
 */
template <typename T>
class SingleLinkedList
{
public:
    SingleLinkedList() = default;
    ~SingleLinkedList() {
        while (head)
        {
            SingleLinkedNode<T>* cur = head;
            head = cur->next;
            delete cur;
        }
    }

    void insert_head(const T& elem) {
        SingleLinkedNode<T>* new_node = new SingleLinkedNode<T>();
        new_node->val = elem;
        new_node->next = head;
        
        if (tail == nullptr)
            head = tail = new_node;
        else
            head = new_node;
    }

    void insert_tail(const T& elem) {
        SingleLinkedNode<T>* new_node = new SingleLinkedNode<T>();
        new_node->val = elem;
        new_node->next = nullptr; 

        if (head == nullptr)
            head = tail = new_node;
        else
        {
            tail->next = new_node;
            tail = new_node;
        }
    }

    void remove(const T& elem) {
        SingleLinkedNode<T>** curr = &head;
        while (*curr)
        {
            SingleLinkedNode<T>* entry = *curr;
            if (entry->val == elem)
            {
                *curr = entry->next;
                delete entry;
            }
            else
            {
                curr = &entry->next;
            }
        }

        if (curr == &head)
        {
            tail = nullptr;
        }
        else
        {
            // 要求Node中next为第一个成员
            tail = (SingleLinkedNode<T>*)curr;
        }
    }

    void revert() {
        SingleLinkedNode<T>* curr = head;
        head = nullptr;
        tail = curr;

        while (curr)
        {
            SingleLinkedNode<T>* entry = curr;
            curr = entry->next;

            entry->next = head;
            head = entry;
        }
    }

    SingleLinkedNode<T>* middle(bool& odd) {
        SingleLinkedNode<T>* pFast = head;
        SingleLinkedNode<T>* pSlow = head;

        odd = false;
        while (pFast)
        {
            pFast = pFast->next;
            if (!pFast)
            {
                odd = true;
                break;
            }

            pFast = pFast->next;
            pSlow = pSlow->next;
        }

        return pSlow;
    }

    T at(int idx) const {
        int i = 0;
        SingleLinkedNode<T>* curr = head;

        while (curr)
        {
            if (i == idx) break;
            curr = curr->next;
            ++i;
        }
        
        assert(i == idx);
        return curr->val;
    }

protected:
    SingleLinkedNode<T>* head {nullptr};
    SingleLinkedNode<T>* tail {nullptr};
};

/**
 * 双向链表结点
 */
template <typename T>
struct DoubleLinkedNode
{
    DoubleLinkedNode<T>* prev;
    DoubleLinkedNode<T>* next;
    T                    val;
};

/**
 * 双向链表，在表头插入元素，在表尾插入元素，
 * 删除指定元素，反转
 */
template <typename T>
class DoubleLinkedList
{
public:
    DoubleLinkedList() = default;
    ~DoubleLinkedList() {
        DoubleLinkedNode<T>* curr = head;
        while(curr)
        {
            DoubleLinkedNode<T>* entry = curr;
            curr = entry->next;
            delete entry;
        }
    }

    void insert_head(const T& elem) {
        DoubleLinkedNode<T>* new_node = new DoubleLinkedNode<T>();
        new_node->prev = nullptr;
        new_node->next = head;
        new_node->val = elem;

        if (head)
            head->prev = new_node;
        else
            tail = new_node;
                
        head = new_node;
    }

    void insert_tail(const T& elem) {
        DoubleLinkedNode<T>* new_node = new DoubleLinkedNode<T>();
        new_node->prev = tail;
        new_node->next = nullptr;
        new_node->val = elem;

        if (tail)
            tail->next = new_node;
        else
            head = new_node;
        
        tail = new_node;        
    }

    void remove(const T& elem) {
        DoubleLinkedNode<T>* curr = head;
        head = nullptr;

        while (curr)
        {
            DoubleLinkedNode<T>* entry = curr;
            if (entry->val == elem)
            {
                if (entry->prev)
                    entry->prev->next = entry->next;
                if (entry->next)
                    entry->next->prev = entry->prev;

                curr = entry->next;
                delete entry;
            }
            else
            {
                if (!head) head = curr;
                tail = curr;
                curr = entry->next;
            }
        }
    }

    void revert() {
        DoubleLinkedNode<T>* curr = head;
        head = tail;
        tail = curr;

        while (curr)
        {
            DoubleLinkedNode<T>* entry = curr;
            curr = entry->next;

            entry->next = entry->prev;
            entry->prev = curr;
        }
    }

    T at(int idx) const {
        int i;
        DoubleLinkedNode<T>* curr;
        std::function<DoubleLinkedNode<T>*(DoubleLinkedNode<T>*, int& )> fn;

        if (idx < 0)
        {
            i = -1;
            curr = tail;
            fn = [](DoubleLinkedNode<T>* node, int& a) { --a; return node->prev; };
        }
        else
        {
            i = 0;
            curr = head;
            fn = [](DoubleLinkedNode<T>* node, int& a) { ++a; return node->next; };
        }
        
        while (curr)
        {
            if (i == idx) break;
            curr = fn(curr, i);
        }
        
        assert(i == idx);
        return curr->val;
    }

    DoubleLinkedNode<T>* head {nullptr};
    DoubleLinkedNode<T>* tail {nullptr};
};

/**
 * 环形链表，表头添加元素，删除指定元素，反转
 */
template <typename T>
class RingLinkedList
{
public:
    RingLinkedList() {
        head = new SingleLinkedNode<T>();
        head->next = head;
    };

    ~RingLinkedList() {
        SingleLinkedNode<T>* curr = head->next;
        while(curr != head)
        {
            SingleLinkedNode<T>* entry = curr;
            curr = entry->next;
            delete entry;
        }

        delete head;
    }

    void insert_head(const T& elem) {
        SingleLinkedNode<T>* new_node = new SingleLinkedNode<T>();
        new_node->val = elem;

        SingleLinkedNode<T>** curr = &head->next;
        SingleLinkedNode<T>* entry = *curr;
        new_node->next = entry;
        *curr = new_node;
    }

    void remove(const T& elem) {
        SingleLinkedNode<T>** curr = &head->next;
        while (*curr != head)
        {
            SingleLinkedNode<T>* entry = *curr;
            if (entry->val == elem)
            {
                *curr = entry->next;
                delete entry;
            }
            else
            {
                curr = &entry->next;
            }
        }
    }

    void revert() {
        SingleLinkedNode<T>* curr = head->next;
        SingleLinkedNode<T>* prev = head;
        while (curr != head)
        {
            SingleLinkedNode<T>* entry = curr;
            curr = entry->next;
            entry->next = prev;
            prev = entry;
        }
        
        head->next = prev;
    }

    T at(int idx) {
        int i = 0;
        SingleLinkedNode<T>* curr = head->next;

        while (curr != head)
        {
            if (i == idx) break;
            curr = curr->next;
            ++i;
        }

        assert(i == idx);
        return curr->val;
    }

    SingleLinkedNode<T>* head;
};


#endif // BOA_LINKED_LIST_LINKED_LIST_H_