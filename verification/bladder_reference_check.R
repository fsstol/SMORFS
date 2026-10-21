## ---------------------------------------------------------------------------
## bladder_reference_check.R
##
## Independent reference implementation and benchmark for the semi-Markov
## bladder cancer model of the paper, written to verify SMORFS and to compare
## its accuracy and computational cost with those of other methods.
##
## The point of this file is that it shares no code and no method with SMORFS.
## SMORFS evaluates the model by Laplace-transforming the renewal system and
## inverting with an accelerated Fourier series. Here the same targets are
## computed three other ways:
##
##   METHOD 1  reference   : the post-onset chain is Markov, so the entrance
##                           density of U1 (a Weibull density) is convolved with
##                           a matrix-exponential kernel. Implemented as an
##                           exponential-midpoint solve of the inhomogeneous
##                           linear ODE, with a Richardson check on the step.
##   METHOD 2  cohort      : the discrete-time cohort approximation a modeler
##                           would build instead, at a user-chosen cycle length.
##   METHOD 3  microsim    : vectorized individual-level simulation.
##
## Outputs: a comparison table (accuracy vs METHOD 1, wall-clock time,
## derivatives available) and a CSV of the reference targets that SMORFS output
## can be diffed against.
##
## IMPORTANT CAVEAT.
## METHOD 1 is exact only because, in the manuscript's example, every post-onset
## transition has a constant hazard: the only non-exponential holding time is in
## the initial state H, where local time coincides with attained age. That is
## what lets the onset density be convolved with a single matrix exponential.
## If any post-onset transition were Weibull, log-normal or Gompertz, this
## reference would not exist and one would be back to nested quadrature, a PDE
## scheme, or the frequency-domain method of the manuscript. Two consequences:
##   (a) the comparison below is fair and usable, because all four methods are
##       computing the same quantity for the same model; but
##   (b) this example does not show the method's full scope: with constant
##       post-onset hazards, the model is a time-inhomogeneous Markov model.
##
## Base R only: no CRAN packages, no compiled code.
## ---------------------------------------------------------------------------

## ===========================================================================
## 1. Model specification
##
## States: H, U1, U2, U3, T1, T2, T3, Dc. Onset H -> U1 has a Weibull hazard in
## local time; because H is the initial state, local time in H is attained age,
## so the entrance density of U1 is exactly the Weibull density. All post-onset
## transitions have constant hazards, so the process restricted to
## {U1,U2,U3,T1,T2,T3,Dc} is a time-homogeneous Markov chain. That is what makes
## an exact reference available for this particular model.
##
## Rates below are the MLE column of Table 2 of the manuscript. Background
## (other-cause) mortality is set to zero throughout: it multiplies every
## quantity by S0(t) and cancels in the instantaneous rates, and holding it at
## zero in all three methods keeps the comparison exact rather than confounding
## it with the manuscript's self-adjusting lambda_0. Set BACKGROUND_HAZARD to a
## function of age to switch it on in all three methods at once.
## ===========================================================================

POST <- c("U1", "U2", "U3", "T1", "T2", "T3", "Dc")   # topological order

onset_shape <- 7.163      # Weibull shape  k
onset_scale <- 139.1      # Weibull scale  lambda

## from -> to -> rate (per year)
rate_list <- list(
  c("U1", "U2", 0.410), c("U1", "T1", 2.531), c("U1", "Dc", 0.012),
  c("U2", "U3", 1.783), c("U2", "T2", 1.932), c("U2", "Dc", 0.049),
  c("U3", "T3", 0.966), c("U3", "Dc", 0.148),
  c("T1", "Dc", 0.010), c("T2", "Dc", 0.067), c("T3", "Dc", 0.142)
)

BACKGROUND_HAZARD <- function(age) rep(0, length(age))

## generator of the post-onset chain
build_Q <- function(rate_list, states = POST) {
  n <- length(states)
  Q <- matrix(0, n, n, dimnames = list(states, states))
  for (tr in rate_list) Q[tr[1], tr[2]] <- as.numeric(tr[3])
  diag(Q) <- -rowSums(Q)
  Q
}
Q <- build_Q(rate_list)

## which transitions are "clinical detection" and which are "cancer death",
## by stage, so that the four target families of Section 4 can be assembled
detect_from  <- c(early = "U1", medium = "U2", late = "U3")
detect_to    <- c(early = "T1", medium = "T2", late = "T3")
preclin_from <- c(early = "U1", medium = "U2", late = "U3")
clin_from    <- c(early = "T1", medium = "T2", late = "T3")

## Weibull onset: entrance density of U1 and survival of H
f_onset <- function(t) (onset_shape / onset_scale) *
  (t / onset_scale)^(onset_shape - 1) * exp(-(t / onset_scale)^onset_shape)
S_onset <- function(t) exp(-(t / onset_scale)^onset_shape)


## ===========================================================================
## 2. Matrix exponential (scaling and squaring with a truncated Taylor series)
##
## expm is not in base R and CRAN may be unavailable, so this is written out.
## For a 7x7 generator with rates below 5 and the step sizes used here, the
## series is machine-accurate at order 12; the scaling loop makes it safe for
## any step.
## ===========================================================================

expm_ss <- function(M, order = 14L) {
  nrm <- max(abs(M))
  s <- if (nrm > 0.5) as.integer(ceiling(log2(nrm / 0.5))) else 0L
  A <- M / 2^s
  E <- diag(nrow(M))
  Term <- E
  for (k in seq_len(order)) {
    Term <- (Term %*% A) / k
    E <- E + Term
  }
  for (i in seq_len(s)) E <- E %*% E
  E
}


## ===========================================================================
## 3. METHOD 1: reference occupancies and targets
##
## p(t) is the vector of expected proportions of the birth cohort in each
## post-onset state at age t, and solves
##
##     dp/dt = Q' p(t) + f_onset(t) e_{U1},        p(0) = 0,
##
## whose solution is the convolution of the Weibull onset density with the
## matrix-exponential kernel. The exponential-midpoint scheme below is second
## order and reuses two precomputed matrices, so the cost is one 7x7 matrix-
## vector product per step. Everything after this is bookkeeping.
## ===========================================================================

reference_solve <- function(Q, h = 1 / 256, t_max = 81) {
  states <- colnames(Q)
  n      <- length(states)
  nstep  <- as.integer(round(t_max / h))
  Qt     <- t(Q)

  E     <- expm_ss(Qt * h)                       # propagator over one step
  src   <- expm_ss(Qt * (h / 2)) %*% (diag(n)[, 1L])   # e_{U1} pushed half a step
  mid   <- (seq_len(nstep) - 0.5) * h            # midpoints
  drive <- h * f_onset(mid)                      # source weight per step

  P <- matrix(0, nstep + 1L, n, dimnames = list(NULL, states))
  p <- numeric(n)
  for (i in seq_len(nstep)) {
    p <- E %*% p + drive[i] * src
    P[i + 1L, ] <- p
  }
  list(t = (0:nstep) * h, P = P, h = h)
}

## assemble the manuscript's target families from an occupancy grid.
## Interval rates follow Section 2.6.2: entries into the state over the
## interval, divided by integrated surviving population over the interval.
assemble_targets <- function(sol, ages = 35:80, per = 1e5) {
  t <- sol$t; P <- sol$P; h <- sol$h
  alive <- S_onset(t) + rowSums(P[, c("U1", "U2", "U3", "T1", "T2", "T3"), drop = FALSE])
  S0    <- exp(-cumsum(BACKGROUND_HAZARD(t)) * h)
  alive <- alive * S0

  ## instantaneous entrance densities of each target event
  flow <- list()
  for (st in names(detect_to))
    flow[[paste0("clin_inc_", st)]] <-
      Q[detect_from[st], detect_to[st]] * P[, detect_from[st]] * S0
  for (st in names(preclin_from))
    flow[[paste0("preclin_mort_", st)]] <-
      Q[preclin_from[st], "Dc"] * P[, preclin_from[st]] * S0
  for (st in names(clin_from))
    flow[[paste0("clin_mort_", st)]] <-
      Q[clin_from[st], "Dc"] * P[, clin_from[st]] * S0
  for (st in names(preclin_from))
    flow[[paste0("preclin_other_", st)]] <-
      BACKGROUND_HAZARD(t) * P[, preclin_from[st]] * S0

  ## trapezoidal integrals on the fine grid, then differenced at the age cuts
  cutidx <- vapply(c(ages, max(ages) + 1), function(a) which.min(abs(t - a)), 1L)
  cum <- function(x) c(0, cumsum((x[-1] + x[-length(x)]) / 2 * h))
  denom <- diff(cum(alive)[cutidx])

  out <- data.frame(age = ages)
  for (nm in names(flow)) out[[nm]] <- per * diff(cum(flow[[nm]])[cutidx]) / denom
  out
}

reference_targets <- function(h = 1 / 256, ages = 35:80)
  assemble_targets(reference_solve(Q, h = h), ages = ages)


## ===========================================================================
## 4. METHOD 2: discrete-time cohort approximation
##
## What a modeler would build instead: one cycle of length delta, rates
## converted to probabilities by 1 - exp(-rate * delta), destinations split in
## proportion to the competing rates, onset probability taken from the Weibull
## cumulative hazard over the cycle. Entries are counted per cycle and the
## denominator is the mid-cycle alive count (the usual half-cycle convention).
## ===========================================================================

cohort_targets <- function(delta = 1 / 12, ages = 35:80, per = 1e5) {
  states <- c("H", POST)
  n      <- length(states)
  nstep  <- as.integer(round((max(ages) + 1) / delta))

  ## one-cycle transition matrix for the post-onset states
  tot  <- -diag(Q)
  Ppost <- matrix(0, length(POST), length(POST), dimnames = list(POST, POST))
  for (i in POST) {
    pleave <- 1 - exp(-tot[i] * delta)
    if (tot[i] > 0) {
      share <- Q[i, ] / tot[i]; share[i] <- 0
      Ppost[i, ] <- pleave * share
    }
    Ppost[i, i] <- 1 - pleave
  }

  occ  <- c(H = 1, setNames(numeric(length(POST)), POST))
  keep <- matrix(0, nstep + 1L, n, dimnames = list(NULL, states))
  ent  <- matrix(0, nstep, length(POST), dimnames = list(NULL, POST))
  keep[1L, ] <- occ
  for (k in seq_len(nstep)) {
    a0 <- (k - 1) * delta; a1 <- k * delta
    ponset <- 1 - exp(-((a1 / onset_scale)^onset_shape -
                        (a0 / onset_scale)^onset_shape))
    newU1 <- occ["H"] * ponset
    moved <- occ[POST] %*% Ppost
    entries <- moved - occ[POST] * diag(Ppost)[POST]   # arrivals from elsewhere
    ent[k, ] <- entries
    ent[k, "U1"] <- ent[k, "U1"] + newU1
    occ["H"]  <- occ["H"] - newU1
    occ[POST] <- as.numeric(moved)
    occ["U1"] <- occ["U1"] + newU1
    keep[k + 1L, ] <- occ
  }

  alive_mid <- rowMeans(cbind(
    rowSums(keep[-nrow(keep), c("H", "U1", "U2", "U3", "T1", "T2", "T3")]),
    rowSums(keep[-1L,          c("H", "U1", "U2", "U3", "T1", "T2", "T3")])))
  cyc_age <- ((seq_len(nstep) - 0.5) * delta)
  bin     <- findInterval(cyc_age, c(ages, max(ages) + 1))

  ## cancer deaths by originating state have to be tracked separately, since
  ## the cohort matrix pools them into Dc
  death_pre  <- sapply(preclin_from, function(s)
    tapply(keep[-nrow(keep), s] * (1 - exp(-tot[s] * delta)) *
             (Q[s, "Dc"] / tot[s]), bin, sum, default = 0)[as.character(seq_along(ages))])
  death_clin <- sapply(clin_from, function(s)
    tapply(keep[-nrow(keep), s] * (1 - exp(-tot[s] * delta)) *
             (Q[s, "Dc"] / tot[s]), bin, sum, default = 0)[as.character(seq_along(ages))])
  detect <- sapply(names(detect_to), function(st)
    tapply(ent[, detect_to[st]], bin, sum, default = 0)[as.character(seq_along(ages))])
  py <- tapply(alive_mid * delta, bin, sum, default = 0)[as.character(seq_along(ages))]

  out <- data.frame(age = ages)
  for (st in colnames(detect))     out[[paste0("clin_inc_", st)]]     <- per * detect[, st] / py
  for (st in colnames(death_pre))  out[[paste0("preclin_mort_", st)]] <- per * death_pre[, st] / py
  for (st in colnames(death_clin)) out[[paste0("clin_mort_", st)]]    <- per * death_clin[, st] / py
  out
}


## ===========================================================================
## 5. METHOD 3: vectorized microsimulation
##
## The model is progressive, so each individual makes at most five jumps and the
## whole cohort can be advanced state by state with vectorized draws: one
## exponential waiting time per individual in a state, one uniform to pick the
## destination in proportion to the competing rates. Person-years are computed
## exactly from each individual's death age rather than by tallying a time grid.
## ===========================================================================

micro_targets <- function(N = 2e6, ages = 35:80, per = 1e5, seed = 1) {
  set.seed(seed)
  tot <- -diag(Q)

  age   <- onset_scale * (-log(runif(N)))^(1 / onset_shape)   # age at onset
  state <- rep("U1", N)
  idx   <- seq_len(N)

  entry <- matrix(NA_real_, N, 3, dimnames = list(NULL, c("T1", "T2", "T3")))
  dead_from <- rep(NA_character_, N)
  dead_age  <- rep(Inf, N)

  living <- c("U1", "U2", "U3", "T1", "T2", "T3")
  repeat {
    active <- idx[state %in% living & is.finite(age)]
    if (!length(active)) break
    moved <- FALSE
    for (s in living) {
      who <- active[state[active] == s]
      if (!length(who)) next
      moved <- TRUE
      age[who] <- age[who] + rexp(length(who), tot[s])
      dest_p <- Q[s, ] / tot[s]; dest_p[s] <- 0
      cp     <- cumsum(dest_p)
      u      <- runif(length(who))
      nxt    <- names(dest_p)[findInterval(u, cp) + 1L]
      to_death <- nxt == "Dc"
      if (any(to_death)) {
        dead_from[who[to_death]] <- s
        dead_age[who[to_death]]  <- age[who[to_death]]
      }
      hit <- nxt %in% c("T1", "T2", "T3")
      if (any(hit)) entry[cbind(who[hit], match(nxt[hit], c("T1", "T2", "T3")))] <-
        age[who[hit]]
      state[who] <- nxt
    }
    if (!moved) break
  }

  cuts <- c(ages, max(ages) + 1)
  ## person-years alive within each age band, exactly
  py <- vapply(seq_along(ages), function(i)
    sum(pmax(0, pmin(dead_age, cuts[i + 1]) - pmin(dead_age, cuts[i]))), 0)

  count <- function(x) as.numeric(table(factor(findInterval(x, cuts),
                                               levels = seq_along(ages))))
  out <- data.frame(age = ages)
  for (k in seq_along(detect_to)) {
    st <- names(detect_to)[k]
    out[[paste0("clin_inc_", st)]] <- per * count(entry[, k]) / py
  }
  for (st in names(preclin_from))
    out[[paste0("preclin_mort_", st)]] <-
      per * count(dead_age[which(dead_from == preclin_from[st])]) / py
  for (st in names(clin_from))
    out[[paste0("clin_mort_", st)]] <-
      per * count(dead_age[which(dead_from == clin_from[st])]) / py
  ## Monte Carlo standard error of each rate, from the event counts
  se <- data.frame(age = ages)
  for (k in seq_along(detect_to)) {
    st <- names(detect_to)[k]
    se[[paste0("clin_inc_", st)]] <- per * sqrt(count(entry[, k])) / py
  }
  list(targets = out, mcse = se, N = N)
}


## ===========================================================================
## 6. Benchmark
## ===========================================================================

max_rel_err <- function(a, b, cols) {
  m <- 0
  for (cc in cols) {
    keep <- is.finite(a[[cc]]) & is.finite(b[[cc]]) & b[[cc]] > 0
    if (any(keep)) m <- max(m, max(abs(a[[cc]][keep] - b[[cc]][keep]) / b[[cc]][keep]))
  }
  m
}

## error broken down by target family, which is far more informative than a
## single worst case: the rare families are where simulation fails.
err_by_family <- function(a, ref) {
  fams <- c(clin_inc = "clin_inc_", clin_mort = "clin_mort_",
            preclin_mort = "preclin_mort_")
  out <- sapply(fams, function(pref) {
    cols <- grep(paste0("^", pref), intersect(names(a), names(ref)), value = TRUE)
    if (!length(cols)) NA_real_ else max_rel_err(a, ref, cols)
  })
  round(out, 4)
}

run_benchmark <- function(N_micro = 2e6, verbose = TRUE) {
  t_ref  <- system.time(ref  <- reference_targets(h = 1 / 256))[["elapsed"]]
  t_ref2 <- system.time(ref2 <- reference_targets(h = 1 / 512))[["elapsed"]]
  tcols  <- setdiff(names(ref), "age")
  richardson <- max_rel_err(ref, ref2, tcols)   # step-halving error estimate

  t_c12 <- system.time(c12 <- cohort_targets(delta = 1 / 12))[["elapsed"]]
  t_c1  <- system.time(c1  <- cohort_targets(delta = 1))[["elapsed"]]
  t_mic <- system.time(mic <- micro_targets(N = N_micro))[["elapsed"]]

  ccols <- intersect(tcols, names(c12))
  tab <- data.frame(
    method = c("Reference (quadrature + matrix exponential, h = 1/256 yr)",
               "Cohort approximation, 1-month cycle",
               "Cohort approximation, 1-year cycle",
               sprintf("Microsimulation, N = %s", format(N_micro, big.mark = ",")),
               "SMORFS (frequency domain)"),
    max_rel_error = c(richardson,
                      max_rel_err(c12, ref, ccols),
                      max_rel_err(c1,  ref, ccols),
                      max_rel_err(mic$targets, ref, ccols),
                      NA_real_),
    seconds = c(t_ref, t_c12, t_c1, t_mic, NA_real_),
    derivatives = c("finite differences only", "finite differences only",
                    "finite differences only", "none (stochastic objective)",
                    "exact, analytic"),
    stringsAsFactors = FALSE
  )

  if (verbose) {
    cat("\n--- Reference targets, clinical incidence per 100,000 ---\n")
    print(ref[ref$age %in% c(40, 50, 60, 70, 80),
              c("age", "clin_inc_early", "clin_inc_medium", "clin_inc_late")],
          row.names = FALSE, digits = 4)
    cat("\n--- Step-halving (Richardson) error of the reference: ",
        signif(richardson, 3), "\n", sep = "")
    cat("\n--- Comparison table ---\n")
    print(tab, row.names = FALSE, digits = 3)
    cat("\nMedian Monte Carlo SE / rate for early clinical incidence, ages 35-50: ",
        signif(median(mic$mcse$clin_inc_early[1:16] /
                      mic$targets$clin_inc_early[1:16], na.rm = TRUE), 3), "\n", sep = "")
  }

  if (verbose) {
    cat("\n--- Max relative error by target family (vs reference) ---\n")
    print(rbind(`cohort 1-month` = err_by_family(c12, ref),
                `cohort 1-year`  = err_by_family(c1,  ref),
                `microsim`       = err_by_family(mic$targets, ref)))
    cat("\n")
  }

  invisible(list(reference = ref, cohort_month = c12, cohort_year = c1,
                 micro = mic, table = tab, richardson = richardson))
}


## ===========================================================================
## 7. Two utilities for comparing SMORFS with the reference
## ===========================================================================

## (a) Diff a SMORFS target export against the reference. The export is a CSV
## with a column `age` and the same target columns as reference_targets.csv,
## computed with background mortality switched off.
compare_smorfs <- function(csv, h = 1 / 256) {
  sm  <- read.csv(csv)
  ref <- reference_targets(h = h, ages = sm$age)
  cols <- setdiff(intersect(names(sm), names(ref)), "age")
  per_col <- sapply(cols, function(cc) max_rel_err(sm[, c("age", cc)],
                                                   ref[, c("age", cc)], cc))
  cat("Columns compared:", length(cols), "\n")
  cat("Max relative deviation, SMORFS vs independent reference:",
      signif(max(per_col), 3), "\n")
  print(round(per_col, 8))
  invisible(per_col)
}

## (b) Independent finite-difference gradient of the Gaussian log-likelihood,
## computed through METHOD 1 rather than through SMORFS. Given SMORFS's
## analytic gradient at the MLE, this checks it with a different evaluator,
## which is a stronger test than checking SMORFS against its own finite
## differences. `targets_csv` needs columns: age, target, rate, se, where
## `target` matches a column name of reference_targets.csv.
theta_mle <- log(c(shape = onset_shape, scale = onset_scale,
                   vapply(rate_list, function(tr) as.numeric(tr[3]), 0)))
names(theta_mle)[-(1:2)] <- vapply(rate_list, function(tr)
  paste0(tr[1], "_", tr[2]), "")

loglik_from_theta <- function(theta, obs, h = 1 / 256) {
  onset_shape <<- exp(theta[1]); onset_scale <<- exp(theta[2])
  rl <- rate_list
  for (i in seq_along(rl)) rl[[i]][3] <- as.character(exp(theta[2 + i]))
  Q <<- build_Q(rl)
  ages <- sort(unique(obs$age))
  pred <- reference_targets(h = h, ages = ages)
  key  <- paste(obs$age, obs$target)
  flat <- data.frame(age = rep(pred$age, ncol(pred) - 1),
                     target = rep(setdiff(names(pred), "age"), each = nrow(pred)),
                     pred = unlist(pred[setdiff(names(pred), "age")]))
  m <- flat$pred[match(key, paste(flat$age, flat$target))]
  sum(-0.5 * ((m - obs$rate) / obs$se)^2 - log(sqrt(2 * pi) * obs$se))
}

fd_gradient <- function(targets_csv, theta = theta_mle, eps = 1e-4) {
  obs <- read.csv(targets_csv)
  keep_shape <- onset_shape; keep_scale <- onset_scale; keep_Q <- Q
  on.exit({onset_shape <<- keep_shape; onset_scale <<- keep_scale; Q <<- keep_Q})
  g <- vapply(seq_along(theta), function(k) {
    tp <- tm <- theta; tp[k] <- tp[k] + eps; tm[k] <- tm[k] - eps
    (loglik_from_theta(tp, obs) - loglik_from_theta(tm, obs)) / (2 * eps)
  }, 0)
  names(g) <- names(theta)
  g
}

## template for the observed-target file, to be filled from the SEER extract
write_target_template <- function(file = "observed_targets_template.csv",
                                  ages = 35:80) {
  fams <- c(outer(c("clin_inc_", "clin_mort_", "preclin_mort_"),
                  c("early", "medium", "late"), paste0))
  write.csv(data.frame(age = rep(ages, length(fams)),
                       target = rep(fams, each = length(ages)),
                       rate = NA_real_, se = NA_real_),
            file, row.names = FALSE)
  cat("Wrote", file, "\n")
}

if (sys.nframe() == 0L) {
  res <- run_benchmark(N_micro = 2e6)
  write.csv(res$reference, "reference_targets.csv", row.names = FALSE)
  write.csv(res$table,     "benchmark_table.csv",   row.names = FALSE)
  cat("\nWrote reference_targets.csv and benchmark_table.csv\n")
  cat("\nTo compare SMORFS with this reference: export SMORFS targets on ages\n",
      "  35:80 with background mortality off, as a CSV with the same column names\n",
      "  as reference_targets.csv, and time its objective, gradient and Hessian\n",
      "  evaluations; then fill row 5 of benchmark_table.csv with\n",
      "  max_rel_err(smorfs, ref). tests/benchmark.py does all of this.\n", sep = "")
}
