#include "unionfind.h"

UnionFind::UnionFind() {}

UnionFind::UnionFind(const UnionFind& other) : parent(other.parent) {}

UnionFind& UnionFind::operator=(const UnionFind& other) {
    if (this != &other) {
        parent = other.parent;
    }
    return *this;
}

std::string UnionFind::make_key(int x, int y) const {
    return std::to_string(x) + "," + std::to_string(y);
}

std::string UnionFind::make_key(int val) const {
    return std::to_string(val);
}

std::string UnionFind::find(const std::string& x) {
    if (parent.find(x) == parent.end()) {
        parent[x] = x;
        return x;
    }
    if (parent[x] != x) {
        parent[x] = find(parent[x]);
    }
    return parent[x];
}

bool UnionFind::join(const std::string& x, const std::string& y) {
    std::string rep_x = find(x);
    std::string rep_y = find(y);
    if (rep_x != rep_y) {
        parent[rep_x] = rep_y;
        return true;
    }
    return false;
}

// 连接两个坐标 (x1,y1) 和 (x2,y2)
bool UnionFind::join_tuple(int x1, int y1, int x2, int y2) {
    std::string key1 = make_key(x1, y1);
    std::string key2 = make_key(x2, y2);
    return join(key1, key2);
}

// 连接边界和坐标
bool UnionFind::join_int_tuple(int edge, int x, int y) {
    std::string s_edge = make_key(edge);
    std::string s_tuple = make_key(x, y);
    return join(s_edge, s_tuple);
}

bool UnionFind::connected_tuple(int x1, int y1, int x2, int y2) {
    std::string key1 = make_key(x1, y1);
    std::string key2 = make_key(x2, y2);
    return find(key1) == find(key2);
}

bool UnionFind::connected_int_tuple(int edge, int x, int y) {
    std::string s_edge = make_key(edge);
    std::string s_tuple = make_key(x, y);
    return find(s_edge) == find(s_tuple);
}