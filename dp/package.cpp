#include "package.h"

using namespace std;

int Package::pack(int capacity, int goods_num,
                  int* goods_weight)
{
    if (goods_num <= 0)
    {
        return 0;
    }
    int 
    
    if (goods_weight[0] == capacity)
    {
        return capacity;
    }
    else if (goods_weight[0] < capacity)
    {
        int remain_capacity = capacity - goods_weight[0];
        goods_num--;
        goods_weight++;

        return max(goods_weight[-1]+pack(remain_capacity, goods_num, goods_weight),
                   pack(capacity, goods_num, goods_weight));
    }
    else
    {
        return pack(capacity, goods_num-1, &goods_weight[1]);
    }
}