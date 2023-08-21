
- [ ] Codes:
  - [x] add update location (possibly we can use the shortest path from networkx).
    - [x] add regret analysis.
    - [x] The current location is actually the measured point (possibly highest variance in dist if the obs_model is disc_max_pt). 
    - [ ] Change the measured loc during path planning.
    then is the calculation of coverage in the regret calculation correct?
  - [x] handle diagonal edges in base_graph.
  - [x] handle boundary for visu in diag_graph.
    - [ ] handle update_boundaries.

- [ ] check params:
  - [ ] obs_model should be disc_max_pt
  - [ ] what is mean value shift

- [ ] Paper:
  - [ ] Submodular optimization cost? 
    - like if the submudular optimization iterates over agents, but not all combinations of queries.
    will this cause additional regret?
