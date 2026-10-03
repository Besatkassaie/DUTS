# santosLarge: how the new attribute/value assignment is picked

Plain-language walkthrough of `experiments/santoslarge_maxd_selection.py` — the algorithm that
replaced santosLarge's random `(protected_attribute, protected_value)` assignment with one chosen to
maximize `|D|`. See `experiments/RESULTS-santoslarge.md` for the resulting numbers and
`experiments/results/santoslarge_maxd_selection_report.csv` for the full before/after per query.

## Setup (once, before touching any table)

Build one lookup for the entire 11,086-table datalake: for every value that appears anywhere in a
categorical column, record which `(table, column)` pairs contain it. This is just bookkeeping done
once up front — cheap, and it's the same lookup the real search already needs (`InvertedIndexOverlap`
's posting-list index), so nothing extra is built purely for this purpose.

## For each of the 78 query tables

1. **Keep the old pick as a safety net.** Note the table's original (random) column + value and the
   real `|D|` it gives. This is the number every candidate has to beat.

2. **List every option the table has.** For every categorical column in this table, and every
   distinct value that appears in it (skipping blank cells), write down `(column, value)` as one
   candidate. A table with 5 categorical columns and 20 values each gives ~100 candidates.

3. **Score every candidate for free.** For each candidate, look up — no searching, just reading a
   number already sitting in the lookup from Setup — how many `(table, column)` pairs *anywhere in
   the corpus* contain that value. A generic value like `"1"` scores in the thousands; a specific
   one-off value (a person's name, an exact date) scores in single digits.

4. **Shortlist the top 20** by that free score.

5. **Actually test the top 20 for real.** For each of those 20, run the true retrieval search (the
   expensive part: semantic similarity + overlap, intersected) and see how big the resulting
   candidate set really is. This step is needed because the free score only measures "does the value
   show up elsewhere" — it doesn't check whether that other column *means the same thing* as this
   one, which the real search does check.

6. **Pick the single best result** — comparing all 20 real results *and* the original from step 1 —
   and that becomes the table's new column + value.

## The two exceptions, handled without any special-case code

- **The same value shows up in two columns of the same table** (e.g. both column 3 and column 7
  contain `"1"`): they're just two separate candidates in step 2, get the identical free score in
  step 3 (the score only cares about the value, not which column it's in), but are tested
  *separately* for real in step 5 — because which column you attach the value to changes the real
  result (different column → different semantic embedding → different intersection with the
  overlap-matched tables). Whichever of the two actually performs better wins.

- **None of the top 20 beat the original**: nothing breaks — step 6 just keeps the original from
  step 1, since it was already in the comparison as the thing to beat. A handful of tables (~4 of 78,
  e.g. `biodiversity_by_county.csv`) landed here — their `|D|` didn't improve at all even after
  checking up to ~200 candidates. That's a legitimate outcome, not a failure: it means the table
  genuinely has no better option available (no other table in the corpus is both semantically similar
  *and* shares any of that table's values).

## One more edge case — decided upstream, before this algorithm ever runs

A table with **zero categorical columns at all** (every column too high-cardinality, domain size
`> theta_cat=50`) never gets an original random pick to begin with, so it never enters the 78-table
list this algorithm operates on in the first place. This is why santosLarge works from 78 of its 80
query tables, not 80 — two tables (`species_scientific_names.csv` and one `...Business Rates
Accounts...` table) have every column's domain size well above 50, so `starmie_fair`'s original
random-assignment generator had nothing to pick from for them, and they were absent from
`protected_attributes_santosLarge.csv` from the start. Not a bug, and not something the maxD
reselection could fix — a genuine structural fact about those two tables.
