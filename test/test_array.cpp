#include "array/array.h"
#include <gtest/gtest.h>

TEST(Array, array)
{
    Array<int> arr(16);
    ASSERT_EQ(arr.size(), 16);

    arr.resize(12);
    ASSERT_EQ(arr.size(), 12);

    arr.resize(24);
    ASSERT_EQ(arr.size(), 24);

    arr[0] = 64;
    arr[2] = 80;

    EXPECT_EQ(arr[0], 64);
    EXPECT_EQ(arr[2], 80);
}

TEST(Array, orderedArray)
{
    OrderedArray<int> arr(16);
    ASSERT_EQ(arr.size(), 0);

    arr.insert(8);
    ASSERT_EQ(arr.size(), 1);
    ASSERT_EQ(arr[0], 8);

    arr.insert(9);
    ASSERT_EQ(arr.size(), 2);
    ASSERT_EQ(arr[0], 8);
    ASSERT_EQ(arr[1], 9);

    arr.insert(3);
    arr.insert(17);
    arr.insert(8);

    ASSERT_EQ(arr.size(), 5);
    ASSERT_EQ(arr[0], 3);
    ASSERT_EQ(arr[1], 8);
    ASSERT_EQ(arr[2], 8);
    ASSERT_EQ(arr[3], 9);
    ASSERT_EQ(arr[4], 17);

    arr.remove(3);
    arr.remove(8);

    ASSERT_EQ(arr.size(), 2);
    ASSERT_EQ(arr[0], 9);
    ASSERT_EQ(arr[1], 17);    
}

TEST(Array, mergeOrderdArray)
{
    OrderedArray<int> arr_0(16);
    ASSERT_EQ(arr_0.size(), 0);

    arr_0.insert(8);
    arr_0.insert(9);
    arr_0.insert(3);
    arr_0.insert(17);
    arr_0.insert(63);

    ASSERT_EQ(arr_0.size(), 5);
    ASSERT_EQ(arr_0[4], 63);

    OrderedArray<int> arr_1(4);
    ASSERT_EQ(arr_1.size(), 0);

    arr_1.insert(4);
    arr_1.insert(2);
    arr_1.insert(1);
    arr_1.insert(3);

    OrderedArray<int> arr_2(16);
    MergeApply<OrderedArray<int>>::merge(arr_0, arr_1, arr_2);
    ASSERT_EQ(arr_2.size(), 9);

    ASSERT_EQ(arr_2[0], 1);
    ASSERT_EQ(arr_2[1], 2);
    ASSERT_EQ(arr_2[2], 3);
    ASSERT_EQ(arr_2[3], 3);
    ASSERT_EQ(arr_2[4], 4);
    ASSERT_EQ(arr_2[5], 8);
    ASSERT_EQ(arr_2[6], 9);
    ASSERT_EQ(arr_2[7], 17);
    ASSERT_EQ(arr_2[8], 63);
}

