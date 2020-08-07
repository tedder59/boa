#ifndef BOA_LINKED_LIST_LINKED_LIST_H_
#define BOA_LINKED_LIST_LINKED_LIST_H_

#include <assert.h>

template <typename T>
struct SingleLinkedNode
{
    SingleLinkedNode<T>* next;
    T                    val;
};

template <typename T>
class SingleLinkedList
{
public:
    void insert_head(const T& elem) {
        SingleLinkedNode<T>* new_node = new SingleLinkedNode<T>();
        new_node->val = elem;
        new_node->next = head;
        head = new_node;
    }

    void erease(const T& elem) {
        for (SingleLinkedNode<T>** curr = &head; *curr;)
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
        SingleLinkedNode<T>* curr = head;
        head = nullptr;
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

    SingleLinkedNode<T>* head {nullptr};
};

template <typename T>
struct DoubleLinkedNode
{
    DoubleLinkedNode<T>* prev;
    DoubleLinkedNode<T>* next;
    T                    val;
};

template <typename T>
class DoubleLinkedList
{
public:
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

    void erease(const T& elem) {
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
        int i = 0;
        DoubleLinkedNode<T>* curr = head;
        if (idx < 0)
        {
            i = -1;
            curr = tail;
        }

        while (curr)
        {
            if (i == idx) break;
            if (idx < 0)
            {
                curr = curr->prev;
                --i;
            }
            else
            {
                curr = curr->next;
                ++i;
            }
        }
        
        assert(i == idx);
        return curr->val;
    }

    DoubleLinkedNode<T>* head {nullptr};
    DoubleLinkedNode<T>* tail {nullptr};
};

template <typename T>
class RingLinkedList
{
public:
    void insert_head(const T& elem) {
        SingleLinkedNode<T>* new_node = new SingleLinkedNode<T>();
        new_node->val = elem;

        if (!head)
        {
            head = new_node;
            head->next = head;
            return;
        }

        SingleLinkedNode<T>* curr = head;
        while(curr->next != head)
        {
            curr = curr->next;
        }

        curr->next = new_node;
        new_node->next = head;
        head = new_node;
    }

    void erease(const T& elem) {
        SingleLinkedNode<T>** curr = &head;
        bool advanced_head = false;

        while (*curr)
        {
            SingleLinkedNode<T>* entry = *curr;
            if (entry->val == elem)
            {
                if (entry == entry->next)
                {
                    head = nullptr;
                    delete entry;
                    return;
                }

                *curr = entry->next;
                delete entry;
            }
            else
            {
                curr = &entry->next;
                advanced_head = true;
            }

            if (advanced_head && *curr == head) break;
        }
    }

    void revert() {
        if (!head) return;

        SingleLinkedNode<T>* prev = head;
        SingleLinkedNode<T>* curr = head;
        while (curr->next != head)
        {
            SingleLinkedNode<T>* entry = curr;
            curr = entry->next;
            entry->next = prev;
            prev = entry;
        }
        
        curr->next = prev;
        head->next = curr;
    }

    T at(int idx) {
        int i = 0;
        SingleLinkedNode<T>* curr = head;

        while (i < idx)
        {
            curr = curr->next;
            ++i;
        }

        assert(i == idx);
        return curr->val;
    }

    SingleLinkedNode<T>* head {nullptr};
};


#endif // BOA_LINKED_LIST_LINKED_LIST_H_