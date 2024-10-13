// head /////////////////////////////////////////////////////////////////////////
#import "@preview/touying:0.5.2": *
#import "@preview/lovelace:0.3.0": *
#import "@local/escher:0.0.0": *
#import "@preview/gviz:0.1.0": * // for rendering dot graphs
#import "@preview/finite:0.3.0": automaton // for rendering automata
#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge, shapes
#import "@preview/equate:0.2.0": equate // <- for numbering equations
#import "@preview/plotst:0.2.0": axis, plot, graph_plot, overlay


#let title = "Monte Carlo Tree Search (MCTS)"
#show: escher-theme.with(
  aspect-ratio: "16-9",
  config-info(author: "Noah Ssssyrkis", date: datetime.today(), title: title),
  config-common(handout: true),
)
#show raw.where(lang: "dot-render"): it => render-image(it.text)
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")

// #let minimax = read("figs/minimax.dot")
// #let alphabeta = read("figs/alphabeta.dot")
#show figure.caption: emph

// body /////////////////////////////////////////////////////////////////////////
#cover-slide()

#focus-slide[
  #set text(size: 23pt)
  "Monte Carlo Tree Search (MCTS) is a method for finding optimal decisions in a given domain by taking random samples in the decision space and building a search tree according to the results."
  #v(2em)
  — #cite(<browne2012>, form: "prose")
]


= Game Tree Search

- Min-Max Tree Search leads to:
  - $alpha - beta$ Pruning leads to:
    - Monte Carlo#footnote[Monte Carlo is fancy word for random] Tree Search (MCTS).
- Min-Max is intuitive, what one would try to do.
- $alpha - beta$ is a smart hack that removes dead-ends.
- When space is too large (see @game_complexity) we simulate and count#footnote[Stretch the power of counting].

#slide[
  - Remember we are in a Markov Decision Processes (MDP):
    - We have a state space $S$.
    - We have an action space $A$.
    - We have a transition function $P$.
    - We have a reward function $R$.
  - We want to find the optimal policy $pi$ (somtimes called $mu$).
  - Actions are chosen by the policy based on the state ($a = pi(s)$).
]

= Minimax

- Determinisic game trees can be fully searched for the best move.
- One intuitively associates game tree search with Min-Max @audibert2009.
- We recursively evaluate the game tree to find the best move.
- We are minimizing the maximum possible gain of the opponent.

#slide[
  #figure(
    kind: "algo",
    supplement: [Algorithm],
    pseudocode-list(stroke: none, booktabs: true, title: "minimax")[
      + function minimax(state, depth, maxim)
        + if depth = 0 or state is terminal; return state value
        + value, cond = maxim ? ($-oo$, max) : ($oo$, min)
        + for each child of state
          + tmp = minimax(child, depth - 1, not maxim)
          + value = cond(value, tmp)
        + return value
    ],
    caption: "Minimax",
  )<minimax>
]

// #focus-slide[
// #figure(
// render-image(minimax),
// caption: [Minimax tree (that can be $alpha-beta$ pruned).],
// )
// ]


= $alpha - beta$ Pruning

- $alpha - beta$ pruning notices that some branches are dead-ends.
- Depending on the order of the branches, we can prune some of them.
- "Some" could be a lot, leading to a huge speedup.

#slide[
  #figure(
    kind: "algo",
    supplement: [Algorithm],
    pseudocode-list(stroke: none, booktabs: true, title: "alphabeta")[
      + function alphabeta(state, depth, $alpha$, $beta$, maxim)
        + if depth = 0 or state is terminal; return state value
        + value, cond = maxim ? ($-oo$, max) : ($oo$, min)
        + for each child of state
          + eval = alphabeta(child, depth - 1, $alpha$, $beta$, not maxim)
          + value = cond(value, eval)
          + $alpha$, $beta$ = maxim ? (max($alpha$, eval), $beta$) : ($alpha$, min($beta$, eval))
          + if $alpha$ <= $beta$ and maxim or $alpha$ >= $beta$ and not maxim; break
        + return value
    ],
    caption: [$alpha - beta$ pruning],
  )<minimax>
]

// #focus-slide[
// #set text(size: 60pt)
// $dots.v$
// ]

#focus-slide[
  #figure(
    table(
      columns: 3,
      inset: 0.5cm,
      stroke: (_, y) => if y > 0 {
        (top: 0.8pt)
      },
      align: center + horizon,
      table.header[*Game*][*Branching factor*][*Game tree*],
      [_Tic-tac-toe_], [4], pause,
      [$10^5$], [_Connect Four_], pause,
      [4], [$10^21$], [$dots.v$],
      [$dots.v$], [$dots.v$], [_Chess_],
      pause, [35], [$10^123$],
      [$dots.v$], [$dots.v$], [$dots.v$],
      [_Go_], pause, [250],
      [$10^360$], [$dots.v$], [$dots.v$],
      [$dots.v$],
    ),
    caption: "Game tree sizes for various games.",
  )<game_complexity>
]



= Multi-Armed Bandits

- The Multi-Armed Bandit#footnote[
Machine in with levers that give rewards.
] Problem is a classic problem in statistics.
- It is suprisingly difficult to solve (skim #cite(<robbins1952>, form: "prose")).
- Suppose we have a slot machine with $k$ arms.
- We want to maximize our reward by pulling the arms.
- We don't know the reward distribution of the arms.
- We need to balance exploration and exploitation.

#slide[
  - Strategy 1: Pull the arms uniformly.
  - Strategy 2: Pull the arm with the highest reward, so far.
    - In the extreme case, we pull one arm, and stick to it.
  - Strategy 3: Pull all ones and then stick to the best.
  - Are strategy 1 and 2 equally good?
  - Is there a mathematical way to balance exploration and exploitation?
]

#slide[
  - Pull the arm with the highest reward, with probability $1 - epsilon$.
  - With probability $epsilon$, pull a random arm ($epsilon$-greedy).
  - Strategy 1 and 2 are special cases where $epsilon = 0$ and $epsilon = 1$ respectively.
  - What is the optimal value of $epsilon$? Is it constant? Discuss.
]

= Upper Confidence Bound

#slide(composer: (5fr, 4fr))[
  - UCB1 (@ucb1) balances exploration and exploitation @auer2002
    - $macron(X_j)$ is the average reward of arm $j$.
    - $n$ is the total number of pulls.
    - $n_j$ is the times arm $j$ was pulled.
  - How do the two terms affect strategy?
][
  $
    "UCB1" = macron(X_j) + sqrt((2  ln n) / n_j)
  $ <ucb1>
  $
    "UCBi" = macron(X_j) / n_j
  $
]

#slide[
  - $macron(X_j)$ encourages exploitation (it is big when the reward is big).
  - $sqrt((2  ln n) / n_j)$ encourages exploration (lowers with increasing $n_j$).
  - Nice statistical properties, that I will spare you.
  - Regret is the difference between ones reward and the optimal reward.
  - UCB1 gets us within a factor of $log(n)$#footnote[#cite(<lai1985>, form: "prose") shows lower regret bound proportional to $O(log(n))$].
]

#slide(composer: (3fr, 2fr))[
  - We try an arm, and see that it is good.
  - We try it again, and become mose sure.
  - At one point, the 90th percentile is better for the "bad" arm#footnote[About which we don't know much yet].
][
  #let x_axis = axis(min: 0, max: 1, step: 1, location: "bottom")
  #let y_axis = axis(min: 0, max: 4, step: 1, location: "left")
  #let pl1 = plot(data: ((0, 1), (1, 1)), axes: (x_axis, y_axis))
  #let g1 = graph_plot(
    pl1,
    (100%, 50%),
    rounding: 30%,
    caption: "Expected rewards",
  )
  #let pl2 = plot(data: ((0.4, 0), (0.7, 3), (1.0, 0)), axes: (x_axis, y_axis))
  #let g2 = graph_plot(pl2, (100%, 100%), rounding: 30%)
  #overlay((g1, g2), (100%, 70%))
]



#focus-slide[
  Randomly sampling trajectories is good.

  Choosing a good $epsilon$ is better.

  UCB1 is even better than that.
]

= Monte Carlo Tree Search

- "Monte Carlo" is statistical lingo for random.
- (MCTS) @browne2012 is a heuristic search algorithm.
- MCTS is state of the art, being used by #cite(<silver2016>, form: "prose") in Go.
- Coding MCTS is Type 3 fun.

// #focus-slide[
// #figure(
// image("figs/simple_mcts.png"),
// caption: [MCTS: a simple idea @browne2012],
// )
// ]

#slide(composer: (auto, 1fr))[
  - Selection: Choose a node to expand.
  - Expansion: Add a child node to the tree.
  - Simulation: Simulate a random playout.
  - Backpropagation: Update node statistics.
][
  #figure(
    kind: "diagram",
    supplement: [Diagram],
    diagram(
      node-stroke: 1pt,
      node-inset: 0.35cm,
      node-corner-radius: 0.2cm,
      node((0, 0), [Selection]),
      edge("->"),
      node((0, 1), [Expansion]),
      edge("->"),
      node((0, 2), [Simulation]),
      edge("->"),
      node((0, 3), [Backpropagation]),
      edge("l,u,u,u,r", "->"),
    ),
  )
]

// #slide[
//   #figure(
//     image("figs/mcts.png"),
//     caption: [MCTS illustration #cite(<browne2012>, form: "prose")],
//   )
// ]

#slide(composer: (4fr, 5fr))[
  - Move down from root until first unexplored node.
  - Expand the node's children accoring to possible moves.
  - Simulate random playout from node with default policy.
  - Backpropagate the result.
][
  //pseudo code
  #figure(
    kind: "algo",
    supplement: [Algorithm],
    pseudocode-list(stroke: none, booktabs: true, title: "mcts")[
      + function mcts($s_0$)
        + create root node $v_0$ with satate $s_0$
        + while within compute budget do
          + $v_l$ = treepolicy($v_0$)
          + $delta$ = defaultpolicy($s(v_l)$)
          + backup($v_l$, $delta$)
        + return $a$(bestchild($v_0$))
    ],
    caption: "MCTS",
  )<mcts>
]

#slide[
  - Alas, what should `bestchild` in @mcts be?
  - Should we sample uniformly?.. or should we use ...
  - UCB1 (or its tree variant UCT)
  - $C_p$ is a constant that balances exploration and exploitation @kocsis2006.
  // - With UCT, MCTS converges to minimax optimal.
][
  $
    "UCT" = macron(X_j) + 2 C p sqrt((2  ln n) / n_j)
  $
]


#slide[
  To sum up:
  - MCTS is a powerful heuristic search algorithm.
  - Many variants exist, with UCT being what brought it glory in Go.
  - It can be pruned, parallelized, and combined with other algorithms.
  - Extremely general: Used in some tree of thought LLM algoritms.
]


// tail /////////////////////////////////////////////////////////////////////////
#set align(top)
#show heading.where(level: 1): set heading(numbering: none)
= References <touying:unoutlined>
#bibliography("library.bib", title: none, style: "ieee")
