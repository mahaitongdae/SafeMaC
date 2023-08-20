
- [ ] Codes:
  - [ ] add update location (possibly we can use the shortest path from networkx).
    - [ ] add regret analysis.
    - [ ] The current location is actually the measured point (possibly highest variance in dist if the obs_model is disc_max_pt). 
    then is the calculation of coverage in the regret calculation correct?
  - [ ] handle diagonal edges in base_graph.
  - [ ] handle boundary for visu in diag_graph.
    - [ ] handle update_boundaries.

- [ ] check params:
  - [ ] obs_model should be disk_center
  - [ ] what is mean value shift

- [ ] Paper:
  - [ ] Submodular optimization cost? 
    - like if the submudular optimization iterates over agents, but not all combinations of queries.
    will this cause additional regret?
