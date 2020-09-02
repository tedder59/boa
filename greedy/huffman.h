#ifndef BOA_GREEDY_HUFFMAN_H_
#define BOA_GREEDY_HUFFMAN_H_

#include <unordered_map>
#include <vector>

struct Code {
    u_char  _bits;
    u_char  _bit_num;

    Code(u_char bits, u_char bit_num)
        : _bits(bits), _bit_num(bit_num) {}
};

class Huffman
{
private:
    
    std::unordered_map<char, Code> _tables;

public:
    Huffman(const std::vector<char>& letters,
            const std::vector<int>& frequences);
    ~Huffman();

    void dump_code_table() const;
};

#endif // BOA_GREEDY_HUFFMAN_H_