#include "huffman.h"
#include <queue>
#include <assert.h>

using namespace std;

struct Node {
    int priority;
    char letter;
    Node(char c, int p) : priority(p), letter(c) {}

    bool operator < (const Node& other) const {
        return priority > other.priority;
    }
};

struct TreeNode {
    TreeNode* left;
    TreeNode* right;
    Node    n;

    TreeNode(const Node& _n) 
        : n(_n.letter, _n.priority) {
        left = nullptr;
        right = nullptr;
    }
};

Huffman::Huffman(const vector<char>& letters,
                 const vector<int>& frequences)
{
    assert(letters.size() == frequences.size());

    priority_queue<Node> queue;
    for (size_t i = 0; i < letters.size(); i++)
    {
        queue.push(Node(letters[i], frequences[i]));
    }
    
    TreeNode* root = nullptr;
    while (!queue.empty())
    {
        Node curr = queue.top();
        TreeNode* leaf = new TreeNode(curr);

        if (!root) 
        {
            root = leaf;
        }
        else 
        {
            int proirity = root->n.priority + leaf->n.priority;
            TreeNode* new_root = new TreeNode(Node(0, proirity));
            if (root->n.priority > curr.priority)
            {
                new_root->left = root;
                new_root->right = leaf;
            }
            else
            {
                new_root->left = leaf;
                new_root->right = root;
            }
            
            root = new_root;
        }

        queue.pop();
    }
    
    u_char curr_bits = 0;
    u_char curr_bit_num = 1;
    while (root)
    {
        TreeNode* curr = root->left;
        if (curr && curr->left)
        {
            _tables.insert({root->right->n.letter,
                            Code(curr_bits & 0x01,
                                 curr_bit_num)});
            delete root->right;
            root = curr;

            curr_bits = curr_bits << 1;
            curr_bit_num++;
        }
        else if (curr && root->right)
        {
            _tables.insert({curr->n.letter,
                            Code(curr_bits, curr_bit_num)});
            delete curr;
            root = root->right;

            curr_bits = curr_bits << 1;
            curr_bit_num++;
            curr_bits &= 0x01;
        }
        else
        {
            _tables.insert({root->n.letter,
                            Code(curr_bits, curr_bit_num)});
            delete root;
            root = nullptr;
        }
    }
}

Huffman::~Huffman()
{

}

void Huffman::dump_code_table() const
{

}