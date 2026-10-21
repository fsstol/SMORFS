#pragma once

class State;
struct tuple_t {
    State* state;
    Risk* risk;
    int_t p_idx;
};

class State {
public:
    std::vector<tuple_t> transitions;
    bool terminal = false;

    // false if this state already has a transition to other: one risk per pair of states
    inline bool add_transition(State* other, Risk* risk) {
        for (const tuple_t& tuple: transitions) if (tuple.state == other) return false;
        transitions.emplace_back(other, risk, p_sum);
        p_sum += risk->n_params;
        return true;
    }

    int_t p_sum = 0;
};

static const State LIFETIME_STATE = State();
static const State SURVIVAL_STATE = State();
static const std::unordered_set<const State*> SPECIAL_STATES = {&SURVIVAL_STATE, &LIFETIME_STATE};

class Model {
public:
    int_t n_params = 0;
    std::unordered_map<Risk*, int_t> risk_indices;
    std::unordered_set<State*> cutoff_states;
    int_t eval_order;

    virtual bool update() = 0;
};
