#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gamestate.h"
#include <random>

namespace py = pybind11;

// ¿ìËÙ rollout º¯Êý
int fast_rollout(const GameState& initial_state) {
    GameState state = initial_state;
    std::random_device rd;
    std::mt19937 gen(rd());

    while (state.winner() == 0) {
        auto legal_moves = state.moves();
        if (legal_moves.empty()) {
            break;
        }

        std::uniform_int_distribution<> dis(0, legal_moves.size() - 1);
        int idx = dis(gen);
        auto move = legal_moves[idx];
        state.play(move.first, move.second);
    }

    return state.winner();
}

PYBIND11_MODULE(hex_cpp, m) {
    m.doc() = "Hex game C++ acceleration module";

    py::class_<GameState>(m, "GameState")
        .def(py::init<int>(), py::arg("size"),
            "Create a new game state with given board size")
        .def(py::init<const GameState&>(), py::arg("other"),
            "Copy constructor")
        .def("get_size", &GameState::get_size,
            "Get board size")
        .def("turn", &GameState::turn,
            "Get current player (1=RED, 2=BLUE)")
        .def("get_board", &GameState::get_board,
            "Get flat board array")
        .def("play", &GameState::play, py::arg("x"), py::arg("y"),
            "Play a move at (x, y)")
        .def("winner", &GameState::winner,
            "Check winner (0=none, 1=RED, 2=BLUE)")
        .def("moves", &GameState::moves,
            "Get list of legal moves as (x,y) tuples")
        .def("neighbors", &GameState::neighbors, py::arg("x"), py::arg("y"),
            "Get neighbors of cell (x, y)");

    m.def("fast_rollout", &fast_rollout, py::arg("state"),
        "Perform a fast random rollout and return the winner");

    m.attr("EMPTY") = 0;
    m.attr("RED") = 1;
    m.attr("BLUE") = 2;
}