// Regression program for the quadrature workspace (test `quadws`, tests/test_quadrature_workspace.py), built with
// AddressSanitizer. The bladder model's structure through SemiMarkov::evaluate: a Hessian at the MLE, then an
// objective at Weibull shape x 0.14, which needs more intervals at a shorter stack. Before the fix, the second call
// wrote past the end of QuadMultiWorkspace::errors (quad_vec.cpp, claim).
#include "../src/src_lib/library.cpp"

int main() {
    std::vector<State*> st(13);
    for (auto& s: st) s = new State;
    for (int i=7; i<13; i++) st[i]->terminal = true;
    SemiMarkov sm;
    for (auto s: st) sm.add_state(s);
    sm.set_initial(st[0]);
    struct Edge {int from, to; double rate;};
    st[0]->add_transition(st[1], new Weibull(7.4, 135.0));
    const Edge edges[] = {{1,2,0.7}, {1,4,4.5}, {2,3,2.5}, {2,5,3.2}, {3,6,1.4}, {4,7,0.009}, {5,8,0.066},
                          {6,9,0.158}, {1,10,0.015}, {2,11,0.065}, {3,12,0.19}};
    for (const auto& e: edges) st[e.from]->add_transition(st[e.to], new Exponential(e.rate));
    if (!sm.update()) return 2;
    std::vector<double> t;
    for (double x=0; x<100; x+=0.07) t.push_back(x);
    const int n = 13, p = sm.n_params;
    std::vector<int_t> cumulative(n+1, 0), cut(n, 0);
    std::vector<double> out((n+2)*(1+p+p*p)*t.size()), timings(5, 0.0), th(p);
    sm.get_parameters(th.data());
    sm.eval_order = 2;
    sm.evaluate(true, cumulative.data(), cut.data(), t.data(), t.size(), 70.0, 30.0, 5, 12, 200.0, out.data(),
                timings.data());
    th[0] += std::log(0.14);
    sm.set_parameters(th.data());
    sm.eval_order = 0;
    sm.evaluate(true, cumulative.data(), cut.data(), t.data(), t.size(), 70.0, 30.0, 5, 12, 200.0, out.data(),
                timings.data());
    printf("done\n");
    return 0;
}
