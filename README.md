# Dunes Flower Defense

An informal proof that generalized Bloons TD games are NP-hard using only Dart Monkeys.
(Allowing Tack Shooters would make the proof a lot more trivial.)

The proof aims to transform an instance of Monotone Rectilinear 3SAT (MR3SAT) into
a playable BTD round with a limited number of placeable towers.
The resulting round is winnable if and only if the original MR3SAT instance is satisfiable.

The transformation will go through these NP-complete problems:

- MR3SAT with incident edges constrained on a square grid
- planar rectilinear vertex cover with edges constrained on a square grid
- planar rectilinear vertex cover with edges constrained on a triangle grid
- planar rectilinear vertex cover with edges of length 1 and vertices constrained on a triangle lattice

None of the listed problems above have been formally proved NP-complete either.
Only MR3SAT has been formally proven in literature.
