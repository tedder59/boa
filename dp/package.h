#ifndef BOA_DP_PACKAGE_H_
#define BOA_DP_PACKAGE_H_

#include <vector>

class Package
{
public:
    static int pack(int capacity, int goods_num,
                    int* goods_weight);
};

#endif