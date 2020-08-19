#ifndef BOA_COMMON_UTILS_H_
#define BOA_COMMON_UTILS_H_

template<typename T>
class MergeApply
{
public:
    static void merge(const T& l0, const T& l1, T& out)
    {
        out = l0 + l1;
    }
};

#endif 