#include "queue/queue.h"
#include <gtest/gtest.h>

TEST(QUEUE, arrayQueue)
{
    ArrayQueue<int> queue(16);
    ASSERT_EQ(queue.empty(), true);
    ASSERT_EQ(queue.full(), false);

    for (int i = 0; i < 5; i++)
    {
        queue.push_back(i);
    }

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(queue.pop_front(), i);
    }

    for (int i = 5; i < 19; i++)
    {
        queue.push_back(i);
    }

    for (int i = 4; i < 19; i++)
    {
        ASSERT_EQ(queue.pop_front(), i);
    }
}

TEST(QUEUE, linkQueue)
{
    LinkQueue<int> queue;

    for (int i = 0; i < 5; i++)
    {
        queue.push_back(i);
    }

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(queue.pop_front(), i);
    }

    for (int i = 5; i < 19; i++)
    {
        queue.push_back(i);
    }

    for (int i = 4; i < 19; i++)
    {
        ASSERT_EQ(queue.pop_front(), i);
    }
}