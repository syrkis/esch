// head /////////////////////////////////////////////////////////////////////////
#import "@preview/touying:0.5.2": *
#import "@preview/lovelace:0.3.0": pseudocode-list
#import "@preview/equate:0.2.0": equate
#import "lib.typ": *

#let title = "| | | | | |"
#show: escher-theme.with(aspect-ratio: "16-9", config-info(author: "Noah Syrkis", date: datetime.today(), title: title))

// body ////////////////////////////////////////////////////////////////////////


#cover-slide()


#focus-slide[
#set text(size: 22pt)
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
- State space is too large to search, so we simulate, counting wins and losses.



= Min-Max Tree Search

- Determinisic game trees can be fully searched for the best move.
- Tic-Tac-Toe has 255,168 possible games (1 sec. on a my mac).
- Chess has $10^120$ possible games (still waiting on my mac)#footnote[The universe will end before this is done.].
- What can we do about this? Discuss.
- One intuitively associates game tree search with Min-Max @audibert2009.
- We recursively evaluate the game tree to find the best move.

#slide[
- Take turns maximizing and minimizing.
- The value of a node is the value of its best child.
- This is exhaustive#footnote[And quickly exhausting.]
][#pseudocode-list(stroke: none, booktabs: true, title: "minmax")[
  + if depth = 0 or node is terminal
    + return heuristic value
  + value = $-oo$ if maxim else $oo$
  + bound = max if maxim else min
  + for each kid of node
    + a = (kid, depth - 1, not maxim)
    + v = minmax(\*a)
    + value = bound(value, v)
  + return value
  ]


]


= $alpha - beta$ Pruning

- $alpha - beta$ pruning is a way to reduce the number of nodes that need to be evaluated in the search tree.
- It is a way to optimize the Min-Max algorithm.

#focus-slide[
#set text(size: 4em, stroke: 2pt)
  – – – –
]

= Multi-Armed Bandit

- The Multi-Armed Bandit Problem is a classic problem in probability theory, statistics, and machine learning.
- It is suprisingly difficult to solve @robbins1952.
- When we pull a lever, we gain both information and reward.
- It is an exploration-exploitation tradeoff.
- I will spare you the math, but suffice it to say, that this problem was solved after the invention of the nuclear bomb.

== UCB1

- Upper Confidence Bound (UCB) @auer2002 is a family of algorithms that seek to balance exploration and exploitation in a multi-armed bandit problem.
- It is used in Monte Carlo Tree Search (MCTS) to determine which nodes to explore.

- MCTS is state of the art, being used by @silver2016 to beat the world champion in Go.

= Monte Carlo Tree Search

- "Monte Carlo" is statistical lingo for random (named after the Mo
- Monte Carlo Tree Search (MCTS) @browne2012 is a heuristic search alg
- It is a best-first search algorithm that builds a tree of nodes by simulating
- Coding MCTS is Type 3 fun (look it up).
- MCTS is state of the art, being used by @silver2016.





#pagebreak()
#set align(top)
#set heading(outlined: false, numbering: none, supplement: none)
= References
#bibliography("library.bib", title: none, style: "ieee")
