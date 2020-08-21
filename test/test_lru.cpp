#include "hash_table/lru.h"
#include <gtest/gtest.h>

TEST(HASH_TABLE, lru)
{
    LRU<Node<int, std::string>> lru(16);
    
    for (int i = 0; i < 23; i++)
    {
        lru.get(i);
    }

    for (int i = 0; i < 3; i++)
    {
        lru.get(i);
    }

    for (int i = 10; i < 27; i++)
    {
        lru.get(i);
    }
}