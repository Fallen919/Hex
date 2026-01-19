#ifndef UNIONFIND_H
#define UNIONFIND_H

#include <string>
#include <unordered_map>

class UnionFind {
private:
    mutable std::unordered_map<std::string, std::string> parent;

    std::string make_key(int x, int y) const;
    std::string make_key(int val) const;

public:
    UnionFind();
    UnionFind(const UnionFind& other);
    UnionFind& operator=(const UnionFind& other);

    std::string find(const std::string& x);
    bool join(const std::string& x, const std::string& y);

    // 连接两个坐标
    bool join_tuple(int x1, int y1, int x2, int y2);

    // 连接边界和坐标
    bool join_int_tuple(int edge, int x, int y);

    bool connected_tuple(int x1, int y1, int x2, int y2);

    bool connected_int_tuple(int edge, int x, int y);
};

#endif // UNIONFIND_H