// Timing test of eval_time_state, outside the library:
//   g++ -o test_main src/src_lib/test_main.cpp -std=c++20 -O3 -march=native [-DSMORFS_USE_BLAS -framework Accelerate]
#include "library.cpp"

int main() {
    double t0 = 4.0/3.0;
    double t1 = 7.0/3.0;
    double dt = 2e-3;
    double rate = 1.5;
    double shape = 3.0;
    double scale = 2.0;

    SemiMarkov sm;
    sm.eval_order = 0;

    Exponential r1 = Exponential(rate);
    Weibull r2 = Weibull(shape, scale);

    State s0, s1, s2;
    s0.add_transition(&s1, &r1);
    s0.add_transition(&s2, &r2);
    printf("transitions=%zu, p=%lld\n", s0.transitions.size(), (long long)s0.p_sum);

    double target[2];
    r1.get_params(target);
    printf("params r1: %f -> %f\n", target[0], exp(target[0]));
    r2.get_params(target);
    printf("params r2: %f -> %f, %f -> %f\n", target[0], exp(target[0]), target[1], exp(target[1]));

    sm.add_state(&s0);
    sm.add_state(&s1);
    sm.add_state(&s2);
    sm.update();

    int_t stack_size = sm.get_stack_size(&s0);
    auto stack = new double[stack_size];
    // a risk's stack: [h, dh, d2h, H, dH, d2H] for up to p_sum parameters (sized as SemiMarkov's workspace does)
    auto stack_risk = new double[2*(1 + s0.p_sum + s0.p_sum*s0.p_sum)];

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<120; i++) sm.eval_time_state(&s0, stack, stack_risk, t0);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::nanoseconds>(stop - start).count();
    printf("evaluations: %f\n", std::ceil((t1-t0)/dt));
    printf("time: %lld ns -> %.6e s\n", (long long)duration, duration/1.0e9);
//    double f = (shape/scale) * pow(t/scale, shape-1) * exp(-pow(t/scale, shape));
//    double S = exp(-pow(t/scale, shape)) * exp(-rate*t);
//    printf("S = %f\n", S);
//    printf("f = %f\n", f);

    delete[] stack;
    delete[] stack_risk;

    return 0;
}
