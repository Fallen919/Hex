#ifndef GAMESTATE_H
#define GAMESTATE_H

#include "unionfind.h"
#include <vector>
#include <utility>

class GameState {
public:
    static const int EMPTY = 0;
    static const int RED = 1;
    static const int BLUE = 2;
    static const int EDGE1 = 1;
    static const int EDGE2 = 2;

private:
    int size;
    int player;
    std::vector<int> board_data;
    UnionFind red_groups;
    UnionFind blue_groups;

    void place_red(int x, int y);
    void place_blue(int x, int y);
    int check_win_red() const;
    int check_win_blue() const;

public:
    GameState(int board_size);
    GameState(const GameState& other);
    GameState& operator=(const GameState& other);

    int get_size() const;
    int turn() const;
    std::vector<int> get_board() const;
    void play(int x, int y);
    int winner() const;
    std::vector<std::pair<int, int>> moves() const;
    std::vector<std::pair<int, int>> neighbors(int x, int y) const;
};

#endif // GAMESTATE_H