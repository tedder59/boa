#include "linked_list/linked_list.h"
#include <gtest/gtest.h>

TEST(LINKED_LIST, singleLinkedList)
{
    SingleLinkedList<int> list;

    list.insert_head(2);
    list.insert_head(4);
    list.insert_head(7);
    list.insert_head(5);

    ASSERT_EQ(list.at(0), 5);
    ASSERT_EQ(list.at(1), 7);
    ASSERT_EQ(list.at(2), 4);
    ASSERT_EQ(list.at(3), 2);

    list.revert();

    ASSERT_EQ(list.at(3), 5);
    ASSERT_EQ(list.at(2), 7);
    ASSERT_EQ(list.at(1), 4);
    ASSERT_EQ(list.at(0), 2);

    bool isOdd;
    SingleLinkedNode<int>* ptr = list.middle(isOdd);
    ASSERT_EQ(ptr->val, 7);
    ASSERT_EQ(isOdd, false);

    list.erease(4);

    ASSERT_EQ(list.at(2), 5);
    ASSERT_EQ(list.at(1), 7);
    ASSERT_EQ(list.at(0), 2);

    ptr = list.middle(isOdd);
    ASSERT_EQ(ptr->val, 7);
    ASSERT_EQ(isOdd, true);
}

TEST(LINKED_LIST, doubleLinkedList)
{
    DoubleLinkedList<int> list;

    list.insert_head(2);
    list.insert_head(4);
    list.insert_head(7);
    list.insert_head(5);

    ASSERT_EQ(list.at(0), 5);
    ASSERT_EQ(list.at(1), 7);
    ASSERT_EQ(list.at(2), 4);
    ASSERT_EQ(list.at(3), 2);

    ASSERT_EQ(list.at(-4), 5);
    ASSERT_EQ(list.at(-3), 7);
    ASSERT_EQ(list.at(-2), 4);
    ASSERT_EQ(list.at(-1), 2);

    list.revert();

    ASSERT_EQ(list.at(3), 5);
    ASSERT_EQ(list.at(2), 7);
    ASSERT_EQ(list.at(1), 4);
    ASSERT_EQ(list.at(0), 2);

    ASSERT_EQ(list.at(-1), 5);
    ASSERT_EQ(list.at(-2), 7);
    ASSERT_EQ(list.at(-3), 4);
    ASSERT_EQ(list.at(-4), 2);

    list.erease(4);

    ASSERT_EQ(list.at(2), 5);
    ASSERT_EQ(list.at(1), 7);
    ASSERT_EQ(list.at(0), 2);

    ASSERT_EQ(list.at(-1), 5);
    ASSERT_EQ(list.at(-2), 7);
    ASSERT_EQ(list.at(-3), 2);
}

TEST(LINKED_LIST, ringLinkedList)
{
    RingLinkedList<int> list;

    list.insert_head(2);
    list.insert_head(4);
    list.insert_head(7);
    list.insert_head(5);

    ASSERT_EQ(list.at(0), 5);
    ASSERT_EQ(list.at(1), 7);
    ASSERT_EQ(list.at(2), 4);
    ASSERT_EQ(list.at(3), 2);
    ASSERT_EQ(list.at(4), 5);
    ASSERT_EQ(list.at(5), 7);
    ASSERT_EQ(list.at(6), 4);
    ASSERT_EQ(list.at(7), 2);

    list.revert();

    ASSERT_EQ(list.at(0), 5);
    ASSERT_EQ(list.at(1), 2);
    ASSERT_EQ(list.at(2), 4);
    ASSERT_EQ(list.at(3), 7);
    ASSERT_EQ(list.at(4), 5);
    
    list.erease(4);

    ASSERT_EQ(list.at(0), 5);
    ASSERT_EQ(list.at(1), 2);
    ASSERT_EQ(list.at(2), 7);
    ASSERT_EQ(list.at(3), 5);
}