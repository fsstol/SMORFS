#pragma once

static const double eps = 1e-15;

typedef Eigen::Matrix<complex_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXcd;

// complex product without the inf/nan handling of std::complex's operator*
static inline complex_t cmul(const complex_t a, const complex_t b) {
    return {a.real()*b.real() - a.imag()*b.imag(), a.real()*b.imag() + a.imag()*b.real()};
}

class SemiMarkov: public Model {
private:
    int_t n_states = 0;
    State* initial = nullptr;

    std::unordered_map<State*, int_t> general_indices;
    std::unordered_map<State*, int_t> exponential_indices;
    int_t n_general = 0;
    int_t n_exponential = 0;

    // States in the order added, and in the model's order (general first, then all-exponential), and the risks in
    // the order of their parameters
    std::vector<State*> added;
    std::vector<State*> order;
    std::vector<Risk*> risks;
    // The structure set_layout saw: the number of states added and of each state's transitions (see structure_current)
    size_t layout_added = 0;
    std::vector<size_t> layout_transitions;

    // Per state: global index of each local parameter (State::p_idx order), and outgoing edges
    // (unique destinations); per edge: source and destination. All set by set_layout.
    std::vector<std::vector<int_t>> local_params;
    std::vector<std::vector<int_t>> state_edges;
    std::vector<int_t> edge_from, edge_to;
    std::vector<int_t> off1, off2, edge_off1, edge_off2;  // offsets of local p and p*p blocks, per state and per edge

    // Strongly connected components of the transition graph, in topological order (every edge between components
    // goes forward): states, position of each state in its component, and the edges into each component from earlier
    // ones; the linear systems are solved component by component
    std::vector<std::vector<int_t>> blocks;
    std::vector<int_t> block_of, block_pos;
    std::vector<std::vector<int_t>> block_in_edges, block_internal_edges;

    // Sparse LU of one component of several states, without pivoting: A = I - L_f is diagonally dominant by columns
    // (each column's off-diagonal |L_f| entries sum to less than 1 when Re gamma > 0), so Gaussian elimination in any
    // order is stable. The order (greedy minimum degree) and the fill-in are found once, and the factorization and the
    // solves are compiled into lists of operations on the entries' values.
    struct SparseLU {
        std::vector<int_t> row;                            // global row of the k-th pivot
        std::vector<int_t> diag_pos, edge_pos;             // value positions of (k, k) and of each internal edge
        int_t n_vals = 0;
        std::vector<int_t> l_start, l_row, l_pos;          // per pivot k: entries (i, k) below it, as (i, position)
        std::vector<int_t> u_start, u_col, u_pos;          // per pivot k: entries (k, j) right of it, as (j, position)
        std::vector<int_t> op_start, op_dst, op_l, op_u;   // per pivot k: value[dst] -= value[l] * value[u]
    };
    std::vector<SparseLU> slu;

    // Transforms at one frequency, in local parameter blocks: survival S, dS, d2S per state; f, df, d2f per edge
    std::vector<complex_t> S, dS, d2S, f, df, d2f;

    // Scratch memory, claimed by update (and grown to the largest frequency count and time grid seen), so that an
    // evaluation allocates nothing
    struct Workspace {
        RowMatrixXcd pi0, pi1, pi2, T;  // right-hand sides, solved in place; row-major for the row updates
        std::vector<complex_t> coef, inv_diag;  // per edge: f, cleaned; per one-state component: 1 / its diagonal entry
        std::vector<std::vector<complex_t>> lu_vals, piv_inv;  // per component of several states: LU values, 1 / pivots
        std::vector<double> stack, stack_risk, h, dh, dH;
        std::vector<complex_t> L_i, row;
        std::vector<double> params, X, R, coeffs;  // parameters of the call; series: transforms (n_stack x 2 terms), result (n_stack x n_t)
        QuadWorkspace quad;
        QuadMultiWorkspace quad_multi;
        std::vector<complex_t> L_general;  // quadrature transforms, per general state: frequencies x stack (max_len)
        int_t max_len = 0;
    } ws;

    // Series bases of the last few time grids: reused only when t (bit for bit), t0 and the Fourier options all match
    struct BasisEntry {
        std::vector<double> t, B;
        double t0 = 0.0, A = 0.0;
        int_t l = 0, periods = 0, last_used = -1;
    };
    static constexpr int_t N_BASIS = 4;
    BasisEntry basis_cache[N_BASIS];
    int_t basis_clock = 0;

    // Per-frequency transforms (pi and the survival transforms) of the last evaluation, reused by the next one when
    // nothing they depend on changed: the parameters, eval_order, the frequencies, the quadrature's limit, the cut-off
    // states and the initial state. Arrival or prevalence, cumulative and terminal flags only affect the assembly.
    struct TransformCache {
        bool valid = false;
        std::vector<double> params;
        std::vector<int_t> cutoff;
        int_t order = -1, terms = 0;
        double gamma_re = 0.0, phase = 0.0, max_time = 0.0;
        const State* initial = nullptr;
        int_t size = 0;  // per frequency: pi0 (n), pi1 (n x p), pi2 (n x ph), S (n), dS, d2S (local blocks)
        std::vector<complex_t> data;
    } tc;

public:
    int_t stats[4] = {0, 0, 0, 0};  // transform computations, transform reuses, basis computations, basis reuses

private:
    const double* get_basis(const double* t, const int_t n_t, const double t0, const double A, const int_t l, const int_t periods) {
        BasisEntry* slot = nullptr;
        for (BasisEntry& e: basis_cache) {
            if (e.last_used >= 0 && (int_t)e.t.size() == n_t && e.t0 == t0 && e.A == A && e.l == l && e.periods == periods
                && std::equal(t, t + n_t, e.t.begin(), [](double a, double b) {return std::memcmp(&a, &b, sizeof(double)) == 0;})) {
                e.last_used = basis_clock++;
                stats[3]++;
                return e.B.data();
            }
            if (slot == nullptr || e.last_used < slot->last_used) slot = &e;  // least recently used
        }
        slot->t.assign(t, t + n_t);
        slot->t0 = t0;
        slot->A = A;
        slot->l = l;
        slot->periods = periods;
        slot->last_used = basis_clock++;
        slot->B.resize(2 * 2*l*periods * n_t);
        if ((int_t)ws.coeffs.size() < 2*periods) ws.coeffs.resize(2*periods);
        fourier_basis(slot->B.data(), ws.coeffs.data(), t, n_t, t0, A, l, periods);
        stats[2]++;
        return slot->B.data();
    }

public:

public:
    std::unordered_map<State*, int_t> state_indices;

    SemiMarkov() = default;

    void add_state(State* state) {
        assert(!SPECIAL_STATES.contains(state));
        if (!state_indices.contains(state)) added.push_back(state);
        state_indices[state] = -1;
    }

    void set_initial(State* state) {
        assert(state_indices.contains(state));
        initial = state;
    }

    // Layout: states with only exponential risks are evaluated analytically, the others ("general") by quadrature;
    // general states come first, each group in the order added. Parameters follow the risks in order of first
    // appearance (states in the order added, transitions in the order added). False if a transition leads to a state
    // not in the model.
    bool update() override {
        risks.clear();
        std::unordered_map<Risk*, int_t> offsets;
        int_t p = 0;
        std::vector<State*> general, exponential;
        for (State* state: added) {
            bool all_exp = true;
            for (const tuple_t& tuple: state->transitions) {
                if (!state_indices.contains(tuple.state)) return false;
                if (!offsets.contains(tuple.risk)) {
                    offsets[tuple.risk] = p;
                    p += tuple.risk->n_params;
                    risks.push_back(tuple.risk);
                }
                all_exp = all_exp && tuple.risk->is_exponential;
            }
            (all_exp ? exponential : general).push_back(state);
        }
        std::vector<State*> states = general;
        states.insert(states.end(), exponential.begin(), exponential.end());
        std::vector<int_t> offs;
        for (Risk* r: risks) offs.push_back(offsets[r]);
        set_layout(states.data(), states.size(), general.size(), risks.data(), offs.data(), risks.size(), p);
        return true;
    }

    // False if a state or a transition was added after the last update: the layout no longer describes the model
    bool structure_current() const {
        if (added.size() != layout_added || order.size() != layout_transitions.size()) return false;
        for (size_t i=0; i<order.size(); i++) if (order[i]->transitions.size() != layout_transitions[i]) return false;
        return true;
    }

    // The parameter vector, in the layout of update
    void set_parameters(const double* params) {
        for (Risk* r: risks) r->set_params(params + risk_indices.at(r));
    }
    void get_parameters(double* params) const {
        for (Risk* r: risks) r->get_params(params + risk_indices.at(r));
    }

    int_t get_n_states() const {return n_states;}
    int_t get_n_general() const {return n_general;}
    int_t get_n_risks() const {return risks.size();}
    const std::vector<State*>& get_order() const {return order;}
    const std::vector<Risk*>& get_risks() const {return risks;}

    // Indices of states and parameters, the edges and the workspace for a given order of states and risks
    void set_layout(State** states, const int_t n, const int_t n_gen, Risk** risks, const int_t* offsets, const int_t n_risks, const int_t p) {
        order.assign(states, states + n);
        layout_added = added.size();
        layout_transitions.resize(n);
        for (int_t i=0; i<n; i++) layout_transitions[i] = states[i]->transitions.size();
        n_states = n;
        n_general = n_gen;
        n_exponential = n - n_gen;
        state_indices.clear();
        general_indices.clear();
        exponential_indices.clear();
        for (int_t i=0; i<n; i++) {
            state_indices[states[i]] = i;
            if (i < n_gen) general_indices[states[i]] = i;
            else exponential_indices[states[i]] = i - n_gen;
        }
        risk_indices.clear();
        for (int_t r=0; r<n_risks; r++) risk_indices[risks[r]] = offsets[r];
        n_params = p;

        local_params.assign(n, {});
        state_edges.assign(n, {});
        edge_from.clear();
        edge_to.clear();
        off1.assign(n+1, 0);
        off2.assign(n+1, 0);
        for (int_t i=0; i<n; i++) {
            for (auto tuple: order[i]->transitions) {
                for (int_t q=0; q<tuple.risk->n_params; q++) local_params[i].push_back(risk_indices.at(tuple.risk) + q);
                const int_t j = state_indices.at(tuple.state);
                bool seen = false;
                for (int_t e: state_edges[i]) seen |= edge_to[e] == j;
                if (!seen) {
                    state_edges[i].push_back(edge_from.size());
                    edge_from.push_back(i);
                    edge_to.push_back(j);
                }
            }
            const int_t pi = local_params[i].size();
            off1[i+1] = off1[i] + pi;
            off2[i+1] = off2[i] + pi*pi;
        }
        const int_t n_edges = edge_from.size();
        edge_off1.assign(n_edges+1, 0);
        edge_off2.assign(n_edges+1, 0);
        for (int_t e=0; e<n_edges; e++) {
            const int_t pi = local_params[edge_from[e]].size();
            edge_off1[e+1] = edge_off1[e] + pi;
            edge_off2[e+1] = edge_off2[e] + pi*pi;
        }
        S.assign(n, 0.0);
        dS.assign(off1[n], 0.0);
        d2S.assign(off2[n], 0.0);
        f.assign(n_edges, 0.0);
        df.assign(edge_off1[n_edges], 0.0);
        d2f.assign(edge_off2[n_edges], 0.0);

        tc.valid = false;

        // workspace for evaluations up to second order
        const int_t ph = p*(p+1)/2;
        strongly_connected_components();
        ws.coef.resize(n_edges);
        ws.inv_diag.assign(blocks.size(), 1.0);
        slu.assign(blocks.size(), SparseLU());
        ws.lu_vals.assign(blocks.size(), {});
        ws.piv_inv.assign(blocks.size(), {});
        for (size_t K=0; K<blocks.size(); K++) {
            if (blocks[K].size() < 2) continue;
            sparse_symbolic(K);
            ws.lu_vals[K].resize(slu[K].n_vals);
            ws.piv_inv[K].resize(blocks[K].size());
        }
        ws.pi0.resize(n, 1);
        ws.pi1.resize(n, p);
        ws.pi2.resize(n, ph);
        ws.T.resize(n, p*p);
        int_t max_len = 0, max_p = 0, max_edges = 0;
        for (int_t i=0; i<n; i++) {
            const int_t pi = local_params[i].size();
            max_p = std::max(max_p, pi);
            max_edges = std::max(max_edges, (int_t)state_edges[i].size());
            max_len = std::max(max_len, (1 + (int_t)order[i]->transitions.size()) * (1 + pi + pi*pi));
        }
        ws.stack.resize(max_len);
        ws.stack_risk.resize(2*(1 + max_p + max_p*max_p));
        ws.L_i.resize(max_len);
        ws.h.resize(max_edges);
        ws.dh.resize(max_edges*max_p);
        ws.dH.resize(max_p);
        ws.row.resize((n+1)*(1 + p + ph));
        ws.quad.reset(max_len);
        ws.max_len = max_len;
    }

    // Tarjan's algorithm, iterative; components come out sinks first, so they are reversed into topological order
    void strongly_connected_components() {
        const int_t n = n_states;
        std::vector<int_t> index(n, -1), low(n, 0), stack, comp_of(n, -1);
        std::vector<char> on_stack(n, 0);
        std::vector<std::vector<int_t>> comps;
        int_t counter = 0;
        std::vector<std::pair<int_t, int_t>> calls;  // (state, next outgoing edge)
        for (int_t root=0; root<n; root++) {
            if (index[root] >= 0) continue;
            calls.push_back({root, 0});
            while (!calls.empty()) {
                auto& [v, next] = calls.back();
                if (next == 0) {
                    index[v] = low[v] = counter++;
                    stack.push_back(v);
                    on_stack[v] = 1;
                }
                bool descended = false;
                while (next < (int_t)state_edges[v].size()) {
                    const int_t w = edge_to[state_edges[v][next++]];
                    if (index[w] < 0) {
                        calls.push_back({w, 0});
                        descended = true;
                        break;
                    }
                    if (on_stack[w]) low[v] = std::min(low[v], index[w]);
                }
                if (descended) continue;
                const int_t vv = v;
                if (low[vv] == index[vv]) {
                    comps.emplace_back();
                    int_t w;
                    do {
                        w = stack.back();
                        stack.pop_back();
                        on_stack[w] = 0;
                        comps.back().push_back(w);
                    } while (w != vv);
                }
                calls.pop_back();
                if (!calls.empty()) low[calls.back().first] = std::min(low[calls.back().first], low[vv]);
            }
        }
        blocks.assign(comps.rbegin(), comps.rend());
        block_of.assign(n, -1);
        block_pos.assign(n, -1);
        for (size_t K=0; K<blocks.size(); K++) {
            for (size_t q=0; q<blocks[K].size(); q++) {
                block_of[blocks[K][q]] = K;
                block_pos[blocks[K][q]] = q;
            }
        }
        block_in_edges.assign(blocks.size(), {});
        block_internal_edges.assign(blocks.size(), {});
        for (int_t e=0; e<(int_t)edge_from.size(); e++) {
            const int_t K = block_of[edge_to[e]];
            (block_of[edge_from[e]] == K ? block_internal_edges[K] : block_in_edges[K]).push_back(e);
        }
    }

    // Elimination order (greedy minimum degree on the symmetrized pattern), fill-in and operation lists of component K
    void sparse_symbolic(const int_t K) {
        const std::vector<int_t>& st = blocks[K];
        const int_t m = st.size();
        std::vector<std::set<int_t>> adj(m);
        for (int_t e: block_internal_edges[K]) {
            const int_t a = block_pos[edge_to[e]], b = block_pos[edge_from[e]];
            if (a != b) {
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        std::vector<int_t> order, inv(m);
        std::vector<char> done(m, 0);
        for (int_t step=0; step<m; step++) {
            int_t v = -1;
            for (int_t q=0; q<m; q++) if (!done[q] && (v < 0 || adj[q].size() < adj[v].size())) v = q;
            done[v] = 1;
            order.push_back(v);
            std::vector<int_t> nb(adj[v].begin(), adj[v].end());
            for (int_t a: nb) {
                adj[a].erase(v);
                for (int_t b: nb) if (a != b) adj[a].insert(b);
            }
            adj[v].clear();
        }
        for (int_t k=0; k<m; k++) inv[order[k]] = k;

        // pattern in elimination order: the diagonal and the internal edges, then the fill-in
        std::vector<std::set<int_t>> rows(m), cols(m);
        auto add = [&](int_t i, int_t j) {rows[i].insert(j); cols[j].insert(i);};
        for (int_t k=0; k<m; k++) add(k, k);
        for (int_t e: block_internal_edges[K]) add(inv[block_pos[edge_to[e]]], inv[block_pos[edge_from[e]]]);
        for (int_t k=0; k<m; k++) {
            std::vector<int_t> L(cols[k].upper_bound(k), cols[k].end()), U(rows[k].upper_bound(k), rows[k].end());
            for (int_t i: L) for (int_t j: U) add(i, j);
        }
        std::map<std::pair<int_t, int_t>, int_t> pos;
        for (int_t i=0; i<m; i++) for (int_t j: rows[i]) pos[{i, j}] = pos.size();

        SparseLU& F = slu[K];
        F = SparseLU();
        F.n_vals = pos.size();
        F.row.resize(m);
        F.diag_pos.resize(m);
        for (int_t k=0; k<m; k++) {
            F.row[k] = st[order[k]];
            F.diag_pos[k] = pos.at({k, k});
        }
        for (int_t e: block_internal_edges[K]) F.edge_pos.push_back(pos.at({inv[block_pos[edge_to[e]]], inv[block_pos[edge_from[e]]]}));
        F.l_start.push_back(0);
        F.u_start.push_back(0);
        F.op_start.push_back(0);
        for (int_t k=0; k<m; k++) {
            std::vector<int_t> L(cols[k].upper_bound(k), cols[k].end()), U(rows[k].upper_bound(k), rows[k].end());
            for (int_t i: L) {
                F.l_row.push_back(i);
                F.l_pos.push_back(pos.at({i, k}));
            }
            for (int_t j: U) {
                F.u_col.push_back(j);
                F.u_pos.push_back(pos.at({k, j}));
            }
            for (int_t i: L) for (int_t j: U) {
                F.op_dst.push_back(pos.at({i, j}));
                F.op_l.push_back(pos.at({i, k}));
                F.op_u.push_back(pos.at({k, j}));
            }
            F.l_start.push_back(F.l_row.size());
            F.u_start.push_back(F.u_col.size());
            F.op_start.push_back(F.op_dst.size());
        }
    }

    // A = I - L_f, entry by entry as v2 cleaned it (non-finite and huge entries clamped), per component: the edges'
    // coefficients for the forward substitution, each one-state component's diagonal entry and each larger component's
    // sparse LU. False if a component is singular.
    bool factor_blocks() {
        auto clean = [](complex_t x) {
            auto c = [](double v) {return std::isnan(v) ? 1e100 : (std::isinf(v) ? (v > 0 ? DBL_MAX : -DBL_MAX) : v);};
            x = complex_t(c(x.real()), c(x.imag()));
            return std::abs(x) > 1e100 ? complex_t(1e100) : x;
        };
        for (int_t e=0; e<(int_t)f.size(); e++) ws.coef[e] = -clean(-f[e]);  // A(to, from) = -f, moved to the right
        for (size_t K=0; K<blocks.size(); K++) {
            const int_t m = blocks[K].size();
            if (m == 1) {
                complex_t diag = 1.0;
                for (int_t e: block_internal_edges[K]) diag = clean(complex_t(1.0) - f[e]);  // a self-transition
                if (diag == complex_t(0.0)) return false;
                ws.inv_diag[K] = 1.0 / diag;
                continue;
            }
            // several states: the sparse LU's values, then its factorization
            const SparseLU& F = slu[K];
            std::vector<complex_t>& v = ws.lu_vals[K];
            std::fill(v.begin(), v.end(), complex_t(0.0));
            for (int_t k=0; k<m; k++) v[F.diag_pos[k]] = 1.0;
            for (size_t q=0; q<block_internal_edges[K].size(); q++) v[F.edge_pos[q]] -= f[block_internal_edges[K][q]];
            for (int_t k=0; k<m; k++) v[F.diag_pos[k]] = clean(v[F.diag_pos[k]]);
            for (size_t q=0; q<F.edge_pos.size(); q++) v[F.edge_pos[q]] = clean(v[F.edge_pos[q]]);
            for (int_t k=0; k<m; k++) {
                const complex_t piv = v[F.diag_pos[k]];
                if (piv == complex_t(0.0)) return false;
                const complex_t pinv = 1.0 / piv;
                ws.piv_inv[K][k] = pinv;
                for (int_t s=F.l_start[k]; s<F.l_start[k+1]; s++) v[F.l_pos[s]] = cmul(v[F.l_pos[s]], pinv);
                for (int_t o=F.op_start[k]; o<F.op_start[k+1]; o++) v[F.op_dst[o]] -= cmul(v[F.op_l[o]], v[F.op_u[o]]);
            }
        }
        return true;
    }

    // Solve (I - L_f) X = X in place (n x r, row-major), component by component in topological order: rows coming
    // from earlier components by forward substitution along the edges, then the component's own system
    void block_solve(RowMatrixXcd& X) {
        for (size_t K=0; K<blocks.size(); K++) {
            for (int_t e: block_in_edges[K]) X.row(edge_to[e]) += ws.coef[e] * X.row(edge_from[e]);
            const std::vector<int_t>& st = blocks[K];
            const int_t m = st.size();
            if (m == 1) {
                if (ws.inv_diag[K] != complex_t(1.0)) X.row(st[0]) *= ws.inv_diag[K];
                continue;
            }
            // several states: L y = b (L unit lower), then U x = y, with the sparse LU
            const SparseLU& F = slu[K];
            const std::vector<complex_t>& v = ws.lu_vals[K];
            for (int_t k=0; k<m; k++) {
                for (int_t s=F.l_start[k]; s<F.l_start[k+1]; s++) X.row(F.row[F.l_row[s]]) -= v[F.l_pos[s]] * X.row(F.row[k]);
            }
            for (int_t k=m-1; k>=0; k--) {
                for (int_t s=F.u_start[k]; s<F.u_start[k+1]; s++) X.row(F.row[k]) -= v[F.u_pos[s]] * X.row(F.row[F.u_col[s]]);
                X.row(F.row[k]) *= ws.piv_inv[K][k];
            }
        }
    }

    int_t edge_index(const int_t i, const int_t j) const {
        for (int_t e: state_edges[i]) if (edge_to[e] == j) return e;
        return -1;
    }
    int_t local_edge(const int_t i, const int_t j) const {
        for (int_t le=0; le<(int_t)state_edges[i].size(); le++) if (edge_to[state_edges[i][le]] == j) return le;
        return -1;
    }

    // Analytic transforms of an all-exponential state i: S = 1/(H + gamma) and, per edge, f = h S, with derivatives
    // with respect to the log-rates (dh = d2h = rate), as v2's Python __eval_laplace_exponentials
    void laplace_exponential_state(const int_t i, const complex_t gamma, const bool cutoff) {
        const int_t pi = local_params[i].size();
        const int_t n_e = state_edges[i].size();
        double* h = ws.h.data();
        double* dh = ws.dh.data();
        double* dH = ws.dH.data();
        std::fill(h, h + n_e, 0.0);
        std::fill(dh, dh + n_e*pi, 0.0);
        std::fill(dH, dH + pi, 0.0);
        if (!cutoff) {
            int_t q = 0;
            for (auto tuple: order[i]->transitions) {
                assert(tuple.risk->is_exponential);
                const int_t e = local_edge(i, state_indices.at(tuple.state));
                const double rate = tuple.risk->rate;
                h[e] += rate;
                dh[e*pi + q] += rate;
                dH[q] += rate;
                q++;
            }
        }
        double H = 0.0;
        for (int_t le=0; le<n_e; le++) H += h[le];
        const complex_t S_i = 1.0 / (H + gamma);
        const complex_t S2 = S_i*S_i;
        const complex_t S3 = S2*S_i;
        S[i] = S_i;
        complex_t* dS_i = dS.data() + off1[i];
        complex_t* d2S_i = d2S.data() + off2[i];
        if (eval_order >= 1) for (int_t q=0; q<pi; q++) dS_i[q] = -dH[q] * S2;
        if (eval_order >= 2) {
            // d2H = diag(dH) for log-rates
            for (int_t q=0; q<pi; q++) {
                for (int_t r=0; r<pi; r++) d2S_i[q*pi + r] = -(q == r ? dH[q] : 0.0) * S2 + (dH[q]*dH[r] + dH[r]*dH[q]) * S3;
            }
        }
        for (int_t le=0; le<n_e; le++) {
            const int_t e = state_edges[i][le];
            f[e] = h[le] * S_i;
            const double* dh_e = dh + le*pi;
            if (eval_order >= 1) {
                complex_t* df_e = df.data() + edge_off1[e];
                for (int_t q=0; q<pi; q++) df_e[q] = dh_e[q] * S_i + h[le] * dS_i[q];
            }
            if (eval_order >= 2) {
                complex_t* d2f_e = d2f.data() + edge_off2[e];
                for (int_t q=0; q<pi; q++) {
                    for (int_t r=0; r<pi; r++) {
                        d2f_e[q*pi + r] = (q == r ? dh_e[q] : 0.0) * S_i + dh_e[q] * dS_i[r] + dh_e[r] * dS_i[q] + h[le] * d2S_i[q*pi + r];
                    }
                }
            }
        }
    }

    // Transforms of the general state i at every frequency, by one shared adaptive quadrature (quad_vec_multi)
    void quadrature_general_state(const int_t i, const complex_t gamma0, const double dgamma, const int_t terms, const double max_time) {
        const State* state = order[i];
        function_t fn = [this, state](double t) {  // small enough for std::function's inline storage
            eval_time_state(state, ws.stack.data(), ws.stack_risk.data(), t);
            return ws.stack.data();
        };
        quad_vec_multi(fn, get_stack_size(state), terms, ws.L_general.data() + i*terms*ws.max_len, gamma0, dgamma, max_time, ws.quad_multi);
    }

    // Transforms of the general state i at frequency k, with the full Hessian over the state's parameters
    void laplace_general_state(const int_t i, const int_t k, const int_t terms) {
        const int_t pi = local_params[i].size();
        const State* state = order[i];
        const int_t len = get_stack_size(state);
        const complex_t* L = ws.L_general.data() + i*terms*ws.max_len + k*len;

        const int_t q1 = eval_order >= 1 ? pi : 0;
        const int_t q2 = eval_order >= 2 ? pi*pi : 0;
        auto place = [&](const complex_t* src, complex_t* v, complex_t* g, complex_t* hh) {
            *v = src[0];
            std::copy(src + 1, src + 1 + q1, g);
            std::copy(src + 1 + q1, src + 1 + q1 + q2, hh);
        };
        place(L, &S[i], dS.data() + off1[i], d2S.data() + off2[i]);
        int_t offset = 1 + q1 + q2;
        for (auto tuple: state->transitions) {
            const int_t e = edge_index(i, state_indices.at(tuple.state));
            place(L + offset, &f[e], df.data() + edge_off1[e], d2f.data() + edge_off2[e]);
            offset += 1 + q1 + q2;
        }
    }

    // Stack of transforms at one frequency, rows (states + lifetime) x (value, gradient p, Hessian p*p): the arrival
    // transforms (prevalence = false) or the prevalence transforms, as v2's Python __get_arrival_transform and
    // __get_prevalence_transform. Returns false if the linear system is singular.
    // Pointers into the cached transforms of frequency k
    struct CachedTransforms {const complex_t *pi0, *pi1, *pi2, *S, *dS, *d2S;};
    CachedTransforms cached(const int_t k) const {
        const int_t n = n_states, p = n_params, ph = p*(p+1)/2;
        const complex_t* base = tc.data.data() + k*tc.size;
        return {base, base + n, base + n + n*p, base + n + n*p + n*ph, base + 2*n + n*p + n*ph, base + 2*n + n*p + n*ph + off1[n]};
    }

    // Transforms at frequency k: the states' S, dS, d2S, and pi = (I - L_f)^-1 e with its derivatives, into the cache.
    // Returns false if the linear system is singular.
    bool base_transforms(const complex_t gamma, const int_t k_freq, const int_t terms, const int_t* cutoff, double* timings) {
        using namespace Eigen;
        const int_t n = n_states;
        const int_t p = n_params;
        const int_t p2 = p*p;
        const int_t ph = p*(p+1)/2;  // Hessian upper triangle, l <= k, row by row

        auto t0 = std::chrono::steady_clock::now();
        for (int_t i=0; i<n_general; i++) {
            if (cutoff[i]) {
                S[i] = 1.0/gamma;
                std::fill(dS.begin() + off1[i], dS.begin() + off1[i+1], 0.0);
                std::fill(d2S.begin() + off2[i], d2S.begin() + off2[i+1], 0.0);
                for (int_t e: state_edges[i]) {
                    f[e] = 0.0;
                    std::fill(df.begin() + edge_off1[e], df.begin() + edge_off1[e+1], 0.0);
                    std::fill(d2f.begin() + edge_off2[e], d2f.begin() + edge_off2[e+1], 0.0);
                }
                continue;
            }
            laplace_general_state(i, k_freq, terms);
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int_t i=n_general; i<n; i++) laplace_exponential_state(i, gamma, cutoff[i]);
        auto t2 = std::chrono::steady_clock::now();

        // A pi = e (A = I - L_f, L_f(a, b) the transform of the density b -> a) and its derivatives, solved component
        // by component (block_solve)
        if (!factor_blocks()) return false;
        RowMatrixXcd& pi0 = ws.pi0;
        RowMatrixXcd& pi1 = ws.pi1;
        RowMatrixXcd& pi2 = ws.pi2;
        pi0.setZero();
        pi0(state_indices.at(initial), 0) = 1.0;
        block_solve(pi0);
        if (eval_order >= 1) {
            // rhs1(:, k) = L_df(:, :, k) pi0: only edges out of the states that own parameter k
            pi1.setZero();
            for (int_t e=0; e<(int_t)f.size(); e++) {
                const int_t b = edge_from[e];
                const complex_t* df_e = df.data() + edge_off1[e];
                for (int_t q=0; q<(int_t)local_params[b].size(); q++) pi1(edge_to[e], local_params[b][q]) += cmul(df_e[q], pi0(b, 0));
            }
            block_solve(pi1);
        }
        if (eval_order >= 2) {
            // rhs2(:, (l, k)) = L_d2f(:, :, l, k) pi0 + T(l, k) + T(k, l), T(l, k) = L_df(:, :, l) pi1(:, k), for l <= k
            RowMatrixXcd& T = ws.T;
            pi2.setZero();
            T.setZero();
            for (int_t e=0; e<(int_t)f.size(); e++) {
                const int_t a = edge_to[e];
                const int_t b = edge_from[e];
                const std::vector<int_t>& g = local_params[b];
                const int_t pb = g.size();
                const complex_t* df_e = df.data() + edge_off1[e];
                const complex_t* d2f_e = d2f.data() + edge_off2[e];
                for (int_t q=0; q<pb; q++) {
                    for (int_t r=0; r<pb; r++) if (g[q] <= g[r]) pi2(a, upper(g[q], g[r])) += cmul(d2f_e[q*pb + r], pi0(b, 0));
                    for (int_t k=0; k<p; k++) T(a, g[q]*p + k) += cmul(df_e[q], pi1(b, k));
                }
            }
            for (int_t a=0; a<n; a++) {
                const complex_t* T_a = T.data() + a*p2;
                complex_t* rhs_a = pi2.data() + a*ph;
                for (int_t l=0; l<p; l++) {
                    for (int_t k=l; k<p; k++) rhs_a[upper(l, k)] += T_a[l*p + k] + T_a[k*p + l];
                }
            }
            block_solve(pi2);
        }
        auto t3 = std::chrono::steady_clock::now();

        const CachedTransforms c = cached(k_freq);
        std::copy(pi0.data(), pi0.data() + n, const_cast<complex_t*>(c.pi0));
        if (eval_order >= 1) std::copy(pi1.data(), pi1.data() + n*p, const_cast<complex_t*>(c.pi1));
        if (eval_order >= 2) std::copy(pi2.data(), pi2.data() + n*ph, const_cast<complex_t*>(c.pi2));
        std::copy(S.begin(), S.end(), const_cast<complex_t*>(c.S));
        if (eval_order >= 1) std::copy(dS.begin(), dS.end(), const_cast<complex_t*>(c.dS));
        if (eval_order >= 2) std::copy(d2S.begin(), d2S.end(), const_cast<complex_t*>(c.d2S));
        timings[0] += std::chrono::duration<double>(t1 - t0).count();
        timings[1] += std::chrono::duration<double>(t2 - t1).count();
        timings[2] += std::chrono::duration<double>(t3 - t2).count();
        return true;
    }

    // Stack of transforms at frequency k from the cache, rows (states + lifetime) x (value, gradient p, Hessian upper
    // triangle): the arrival transforms (prevalence = false), divided by gamma in the cumulative rows, or the prevalence
    // transforms
    void assemble(const complex_t gamma, const int_t k_freq, const bool prevalence, const int_t* cumulative, complex_t* out,
                  double* timings) {
        using namespace Eigen;
        auto t3 = std::chrono::steady_clock::now();
        const int_t n = n_states;
        const int_t p = n_params;
        const int_t ph = p*(p+1)/2;
        const int_t height = get_height();
        const CachedTransforms c = cached(k_freq);
        Map<const VectorXcd> pi0(c.pi0, n);
        Map<const RowMatrixXcd> pi1(c.pi1, n, p);
        Map<const RowMatrixXcd> pi2(c.pi2, n, ph);
        const complex_t* S = c.S;
        const complex_t* dS = c.dS;
        const complex_t* d2S = c.d2S;

        // (written out: std::complex's operator* calls __muldc3 for C99's inf/nan rules, which also blocks vectorization)
        for (int_t i=0; i<n; i++) {
            complex_t* row = out + i*height;
            if (!prevalence) {
                row[0] = pi0(i);
                if (eval_order >= 1) for (int_t k=0; k<p; k++) row[1+k] = pi1(i, k);
                if (eval_order >= 2) for (int_t lk=0; lk<ph; lk++) row[1+p+lk] = pi2(i, lk);
                continue;
            }
            // prevalence: pi S and its derivatives; dS and d2S are non-zero only in the state's own parameters
            const complex_t Si = S[i];
            const std::vector<int_t>& g = local_params[i];
            const int_t pi = g.size();
            const complex_t* dS_i = dS + off1[i];
            const complex_t* d2S_i = d2S + off2[i];
            row[0] = cmul(pi0(i), Si);
            if (eval_order >= 1) {
                for (int_t k=0; k<p; k++) row[1+k] = cmul(Si, pi1(i, k));
                for (int_t q=0; q<pi; q++) row[1+g[q]] += cmul(pi0(i), dS_i[q]);
            }
            if (eval_order >= 2) {
                complex_t* hess = row + 1 + p;
                for (int_t lk=0; lk<ph; lk++) hess[lk] = cmul(pi2(i, lk), Si);
                // pi1_l dS_k + pi1_k dS_l: for an own parameter a, pi1_m dS_a enters the pair {a, m} (twice when m = a)
                for (int_t q=0; q<pi; q++) {
                    const int_t a = g[q];
                    for (int_t m=0; m<p; m++) {
                        const complex_t term = cmul(pi1(i, m), dS_i[q]);
                        hess[upper(std::min(a, m), std::max(a, m))] += m == a ? term + term : term;
                    }
                }
                for (int_t q=0; q<pi; q++) {
                    for (int_t r=0; r<pi; r++) if (g[q] <= g[r]) hess[upper(g[q], g[r])] += cmul(pi0(i), d2S_i[q*pi + r]);
                }
            }
        }
        // lifetime row: total arrival in the terminal states (cumulative for the prevalence)
        complex_t* life = out + n*height;
        std::fill(life, life + height, complex_t(0.0));
        for (int_t i=0; i<n; i++) {
            if (!order[i]->terminal) continue;
            life[0] += pi0(i);
            if (eval_order >= 1) for (int_t k=0; k<p; k++) life[1+k] += pi1(i, k);
            if (eval_order >= 2) for (int_t lk=0; lk<ph; lk++) life[1+p+lk] += pi2(i, lk);
        }
        const complex_t inv_gamma = 1.0 / gamma;
        if (prevalence) for (int_t h=0; h<height; h++) life[h] = cmul(life[h], inv_gamma);
        else {
            for (int_t i=0; i<=n; i++) {
                if (cumulative[i]) for (int_t h=0; h<height; h++) out[i*height + h] = cmul(out[i*height + h], inv_gamma);
            }
        }
        timings[3] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t3).count();
    }

    // Entries per stack row: value, gradient (p) and the Hessian's upper triangle (p(p+1)/2)
    int_t get_height() const {
        return 1 + (eval_order >= 1 ? n_params : 0) + (eval_order >= 2 ? n_params*(n_params+1)/2 : 0);
    }
    int_t upper(const int_t l, const int_t k) const {return l*n_params - l*(l-1)/2 + (k - l);}

    // Arrival (prevalence = false) or prevalence at times t (sorted): transforms at the 2 l periods frequencies, the
    // accelerated Fourier series, and the Hessian expanded to p x p. out: (n+2) x (1 + p + p*p) x n_t, as far as
    // eval_order goes: the states, the lifetime (arrival in the terminal states) and the survival (1 - lifetime);
    // states and lifetime are zero at t <= 0. Returns false if a linear system is singular.
    bool evaluate(const bool prevalence, const int_t* cumulative, const int_t* cutoff, const double* t, const int_t n_t, const double t0, const double A, const int_t l, const int_t periods,
                  const double max_time, double* out, double* timings) {
        using namespace Eigen;
        if (!structure_current()) return false;
        const int_t n_stack = (n_states + 1) * get_height();
        const int_t terms = 2 * l * periods;
        const double gamma_re = A / (2 * l * t0);
        const double phase = M_PI / (l * t0);
        if ((int_t)ws.X.size() < n_stack*2*terms) ws.X.resize(n_stack*2*terms);
        Map<MatrixXd> X(ws.X.data(), n_stack, 2*terms);

        // the transforms at every frequency, unless the cache holds them for the same inputs
        if ((int_t)ws.params.size() < n_params) ws.params.resize(n_params);
        get_parameters(ws.params.data());
        const bool reuse = tc.valid && tc.order == eval_order && tc.terms == terms && tc.gamma_re == gamma_re
            && tc.phase == phase && tc.max_time == max_time && tc.initial == initial
            && std::equal(cutoff, cutoff + n_states, tc.cutoff.begin())
            && std::equal(ws.params.begin(), ws.params.begin() + n_params, tc.params.begin(),
                          [](double a, double b) {return std::memcmp(&a, &b, sizeof(double)) == 0;});
        if (reuse) stats[1]++;
        else {
            tc.valid = false;
            const int_t p = n_params, ph = p*(p+1)/2;
            tc.size = 2*n_states + n_states*p + n_states*ph + off1[n_states] + off2[n_states];
            if ((int_t)tc.data.size() < terms*tc.size) tc.data.resize(terms*tc.size);
            // general states: one quadrature for all frequencies
            auto tq = std::chrono::steady_clock::now();
            if ((int_t)ws.L_general.size() < n_general*terms*ws.max_len) ws.L_general.resize(n_general*terms*ws.max_len);
            for (int_t i=0; i<n_general; i++) if (!cutoff[i]) quadrature_general_state(i, complex_t(gamma_re, 0.0), phase, terms, max_time);
            timings[0] += std::chrono::duration<double>(std::chrono::steady_clock::now() - tq).count();
            for (int_t k=0; k<terms; k++) {
                if (!base_transforms(complex_t(gamma_re, k * phase), k, terms, cutoff, timings)) return false;
            }
            tc.params.assign(ws.params.begin(), ws.params.begin() + n_params);
            tc.cutoff.assign(cutoff, cutoff + n_states);
            tc.order = eval_order;
            tc.terms = terms;
            tc.gamma_re = gamma_re;
            tc.phase = phase;
            tc.max_time = max_time;
            tc.initial = initial;
            tc.valid = true;
            stats[0]++;
        }

        for (int_t k=0; k<terms; k++) {
            assemble(complex_t(gamma_re, k * phase), k, prevalence, cumulative, ws.row.data(), timings);
            for (int_t s=0; s<n_stack; s++) {
                X(s, k) = ws.row[s].real();
                X(s, terms + k) = ws.row[s].imag();
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        const double* B = get_basis(t, n_t, t0, A, l, periods);
        const int_t p = n_params;
        double* res = out;
        if (eval_order >= 2) {
            if ((int_t)ws.R.size() < n_stack*n_t) ws.R.resize(n_stack*n_t);
            res = ws.R.data();
        }
        series_product(res, X.data(), B, n_stack, 2*terms, n_t);
        if (eval_order >= 2) expand_hessian(res, out, n_states + 1, p, n_t);

        const int_t idx0 = std::upper_bound(t, t + n_t, 0.0) - t;
        const int_t h_full = 1 + (eval_order >= 1 ? p : 0) + (eval_order >= 2 ? p*p : 0);
        const int_t n_rows = (n_states + 1) * h_full;
        for (int_t r=0; r<n_rows; r++) std::fill(out + r*n_t, out + r*n_t + idx0, 0.0);

        // survival row: 1 - lifetime, and the lifetime's derivatives negated
        const double* life = out + n_states*h_full*n_t;
        double* surv = out + (n_states + 1)*h_full*n_t;
        for (int_t i=0; i<n_t; i++) surv[i] = 1.0 - life[i];
        for (int_t j=n_t; j<h_full*n_t; j++) surv[j] = -life[j];
        timings[4] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        return true;
    }

    int_t get_stack_size(const State* state) {
        // Target stack layout: [[S_i, dS_i, d2S_i], [f_ij, df_ij, d2f_ij] over j]
        // Sizes: [[1, p, p*p], [1, p, p*p] over j]
        // Where p_ij = number of parameters for risk i->j, p = sum_j p_ij
        const int_t p = state->p_sum;
        int_t stack_size = 1;
        if (eval_order >= 1) stack_size += p;
        if (eval_order >= 2) stack_size += p*p;

        stack_size *= (1+state->transitions.size());
        return stack_size;
    }

    void eval_time_state(const State* i, double* SMORFS_RESTRICT stack_base, double* SMORFS_RESTRICT stack_risk, const double t) {
        // TODO: speed up using matrix library
        // Target stack layout: [[S_i, dS_i, d2S_i], [f_ij, df_ij, d2f_ij] over j]
        // Sizes: [[1, p, p*p], [1, p, p*p] over j], total (1+transitions)*(1+p+p*p)
        // Where p_ij = number of parameters for risk i->j, p = sum_j p_ij

        // Start by evaluating doffset = 1+p+p*p
        const int_t p = i->p_sum;
        int_t doffset = 1;
        if (eval_order >= 1) doffset += p;
        if (eval_order >= 2) doffset += p*p;

        // Initially the stack is zeroed out
        const int_t len_j = i->transitions.size();
        std::fill(stack_base,stack_base+(1+len_j)*doffset,0.0);

        double* SMORFS_RESTRICT stack_i = stack_base;
        double* SMORFS_RESTRICT stack_ij = stack_base + doffset;

        // Construct initial stack layout: [[H_i, dH_i, d2H_i], [h_ij, dh_ij, d2h_ij] over j]
        // Sizes: [[1, p, p], [1, p, p*p] over j], total (1+transitions)*(1+p+p*p)
        for (auto tuple: i->transitions) {
            // Start by evaluating risk function
            const Risk* risk = tuple.risk;
            risk->eval_time(t, eval_order, stack_risk);

            const int_t risk_idx = tuple.p_idx;
            const int_t p_ij = risk->n_params;
            const int_t p2_ij = p_ij*p_ij;
            // stack_risk layout [[h_ij, dh_ij, d2h_ij], [H_ij, dH_ij, d2H_ij]], sizes [[1, p_ij, p_ij*p_ij], [1, p_ij, p_ij*p_ij]]
            const int_t offset_ij = 1+p_ij+p_ij*p_ij;

            stack_ij[0] = stack_risk[0];  // copy h_ij
            stack_i[0] += stack_risk[offset_ij];  // accumulate H_ij on H_i
            if (eval_order >= 1) {
                for (int_t idx=0; idx<p_ij; idx++) {
                    stack_ij[1+risk_idx+idx] = stack_risk[1+idx];  // copy dh_ij
                    stack_i[1+risk_idx+idx] += stack_risk[offset_ij+1+idx];  // accumulate dH_ij on dH_i
                }
            }
            if (eval_order >= 2) {
                for (int_t idx=0; idx<p2_ij; idx++) {
                    const int_t row = (idx / p_ij) + risk_idx;
                    const int_t col = (idx % p_ij) + risk_idx;
                    const int_t offset1 = 1+p+p*row+col;
                    const int_t offset2 = 1+p_ij+idx;
                    stack_ij[offset1] = stack_risk[offset2];  // copy dh_ij
                    stack_i[offset1] += stack_risk[offset_ij+offset2];  // accumulate d2H_ij on d2H_i
                }
            }
            stack_ij += doffset;
        }

        // Current stack layout: [[H_i, dH_i, d2H_i], [h_ij, dh_ij, d2h_ij] over j]
        // Sizes: [[1, p, p], [1, p, p*p] over j], total (1+transitions)*(1+p+p*p)
        // New first part:
        // [S_i, dS_i, d2S_i] = [exp(-H_i), -dH_i * exp(-H_i), -d2H_i * exp(-H_i) + dH_i * dH_i^T * exp(-H_i)]
        //                     = S_i * [1, -dH_i, -d2H_i + dH_i * dH_i^T]
        const double S_i = exp(-stack_i[0]);
        if (eval_order >= 2) {
            const int_t p2 = p*p;
            for (int_t idx=0; idx<p2; idx++) {
                const int_t row = idx / p;
                const int_t col = idx % p;
                stack_i[1+p+idx] = (-stack_i[1+p+idx] + stack_i[1+row]*stack_i[1+col]) * S_i;
            }
        }
        if (eval_order >= 1) {
            for (int_t idx=0; idx< p; idx++) {
                stack_i[1+idx] = -stack_i[1+idx] * S_i;  // dS_i = -dH_i * S_i
            }
        }
        stack_i[0] = S_i;

        // New second part:
        // [f_ij, df_ij, d2f_ij] = [
        //      h_ij * S_i,
        //      dh_ij * S_i + h_ij * dS_i,
        //      d2h_ij * S_i + sym(dh_ij, dS_i) + h_ij * d2S_i
        // ]
        // where for vectors v, w, sym(v, w) = v * w^T + w * v^T
        stack_ij = stack_base + doffset;
        for (int_t j=0; j<len_j; j++) {
            if (eval_order >= 2) {
                const int_t p2 = p*p;
                for (int_t idx=0; idx<p2; idx++) {
                    const int_t row = idx / p;
                    const int_t col = idx % p;
                    const double sym = stack_ij[1+row]*stack_i[1+col] + stack_ij[1+col]*stack_i[1+row];
                    stack_ij[1+p+idx] = stack_ij[1+p+idx] * S_i + sym + stack_ij[0] * stack_i[1+p+idx];
                }
            }
            if (eval_order >= 1) {
                for (int_t idx=0; idx< p; idx++) {
                    stack_ij[1+idx] = stack_ij[1+idx] * S_i + stack_ij[0] * stack_i[1+idx];
                }
            }
            stack_ij[0] = stack_ij[0] * S_i;

            stack_ij += doffset;
        }
    }

    void eval_integrals(complex_t gamma, double* SMORFS_RESTRICT stack, double* SMORFS_RESTRICT stack_risk, complex_t* stack_c) {
        // Find a common size to fit the stacks of all states
        int_t stack_size = 0;
        int_t* stack_sizes = new int_t[general_indices.size()];
        for (auto pair: general_indices) {
            State* state = pair.first;
            int_t idx = pair.second;
            stack_sizes[idx] = get_stack_size(state);
            stack_size = std::max(stack_size, stack_sizes[idx]);
        }

        for (auto pair: general_indices) {
            std::fill(stack_c,stack_c+stack_size,0.0);

            State* i = pair.first;
            int_t idx = pair.second;
            int_t len = stack_sizes[idx];
            function_t f = [&](double t) {
                eval_time_state(i, stack, stack_risk, t);
                return stack;
            };
//            if (gamma.imag() == 0) intdei1(f, len, 0.0, eps, stack_c, nullptr);
//            else intdeo1(f, len, 0.0, gamma, eps, stack_c, nullptr);
        }

        delete[] stack_sizes;
    }
};
