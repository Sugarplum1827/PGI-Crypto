# maximal_gap_records_1e8.csv

The 25 maximal prime gaps (records) with start prime <= 1e8: OEIS A002386 (start prime),
A005250 (gap), A005669 (1-based index of the start prime).  Source: `oeis_validation.csv`
of the frozen project (C++ output == OEIS for all 25 rows, status PASS).  Used by
`pgi_crypto.labels.MaximalGapCatalogue` for the catalogue-based MGV rule
(every window containing a record gap, +/- 5 windows), which reproduces all 185 MGV
windows of the frozen dataset exactly (see tests/test_labels.py).
