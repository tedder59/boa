#include "stack/stack.h"
#include <gtest/gtest.h>

TEST(STACK, arrayStack)
{
    ArrayStack<int> stack(16);

    for (int i = 0; i < 5; i++)
    {
        stack.push(i);
    }

    for (int i = 4; i > 0; --i)
    {
        ASSERT_EQ(stack.pop(), i);
    }
}

TEST(STACK, linkStack)
{
    LinkStack<int> stack;

    for (int i = 0; i < 5; i++)
    {
        stack.push(i);
    }

    for (int i = 4; i > 0; --i)
    {
        ASSERT_EQ(stack.pop(), i);
    }
}