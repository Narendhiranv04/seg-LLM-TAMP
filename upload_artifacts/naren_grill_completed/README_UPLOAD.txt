Current snapshot source commit: 8cbfc8e6
Archive parts: 7
Rebuild:
  cat TAMP-PDDL.tar.zst.part-* > TAMP-PDDL.tar.zst
  zstd -d -c TAMP-PDDL.tar.zst | tar -xf -
