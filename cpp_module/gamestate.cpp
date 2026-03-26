#include "gamestate.h"
#include <algorithm>
#include <stdexcept>

GameState::GameState(int board_size)
    : size(board_size), player(RED), board_data(board_size* board_size, EMPTY) {}

GameState::GameState(const GameState& other)
    : size(other.size), player(other.player), board_data(other.board_data),
    red_groups(other.red_groups), blue_groups(other.blue_groups) {}

GameState& GameState::operator=(const GameState& other) {
    if (this != &other) {
        size = other.size;
        player = other.player;
        board_data = other.board_data;
        red_groups = other.red_groups;
        blue_groups = other.blue_groups;
    }
    return *this;
}

int GameState::get_size() const {
    return size;
}

int GameState::turn() const {
    return player;
}

std::vector<int> GameState::get_board() const {
    return board_data;
}

void GameState::place_red(int x, int y) {
    // 连接到边界（x方向：上/下）
    if (x == 0) {
        red_groups.join_int_tuple(EDGE1, x, y);
    }
    if (x == size - 1) {
        red_groups.join_int_tuple(EDGE2, x, y);
    }

    // ���ӵ����ڵĺ�ɫ���� - ʹ�� join_tuple
    auto neighs = neighbors(x, y);
    for (const auto& n : neighs) {
        if (board_data[n.first * size + n.second] == RED) {
            red_groups.join_tuple(x, y, n.first, n.second);
        }
    }
}

void GameState::place_blue(int x, int y) {
    // 连接到边界（y方向：左/右）
    if (y == 0) {
        blue_groups.join_int_tuple(EDGE1, x, y);
    }
    if (y == size - 1) {
        blue_groups.join_int_tuple(EDGE2, x, y);
    }

    // ���ӵ����ڵ���ɫ���� - ʹ�� join_tuple
    auto neighs = neighbors(x, y);
    for (const auto& n : neighs) {
        if (board_data[n.first * size + n.second] == BLUE) {
            blue_groups.join_tuple(x, y, n.first, n.second);
        }
    }
}

void GameState::play(int x, int y) {
    if (x < 0 || x >= size || y < 0 || y >= size) {
        throw std::invalid_argument("Move out of bounds");
    }

    int idx = x * size + y;
    if (board_data[idx] != EMPTY) {
        throw std::invalid_argument("Cell already occupied");
    }

    board_data[idx] = player;

    if (player == RED) {
        place_red(x, y);
        player = BLUE;
    }
    else {
        place_blue(x, y);
        player = RED;
    }
}

int GameState::check_win_red() const {
    UnionFind& rg = const_cast<UnionFind&>(red_groups);
    std::string edge1_key = std::to_string(EDGE1);
    std::string edge2_key = std::to_string(EDGE2);
    return rg.find(edge1_key) == rg.find(edge2_key) ? RED : 0;
}

int GameState::check_win_blue() const {
    UnionFind& bg = const_cast<UnionFind&>(blue_groups);
    std::string edge1_key = std::to_string(EDGE1);
    std::string edge2_key = std::to_string(EDGE2);
    return bg.find(edge1_key) == bg.find(edge2_key) ? BLUE : 0;
}
int GameState::winner() const {
    int red_win = check_win_red();
    if (red_win) return red_win;
    return check_win_blue();
}

std::vector<std::pair<int, int>> GameState::moves() const {
    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (board_data[i * size + j] == EMPTY) {
                result.push_back({ i, j });
            }
        }
    }
    return result;
}

std::vector<std::pair<int, int>> GameState::neighbors(int x, int y) const {
    std::vector<std::pair<int, int>> result;

    // ����������ھ� (hex grid)
    int dx[] = { -1, -1, 0, 0, 1, 1 };
    int dy[] = { 0, 1, -1, 1, -1, 0 };

    for (int i = 0; i < 6; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
            result.push_back({ nx, ny });
        }
    }

    return result;
}