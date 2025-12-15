Program for analyzing concept inventory data in CSV form, exported directly from Qualtrics.

Currently supported surveys are EMCS, BEMA, and EBAPS

Configuration lines are 6-8

Uses fuzzy matching to parse and match unique identifiers in the data and matches responses against hard-coded correct responses 

Throws out responses which don't pass an attention check in each survey

Reports pre, post, and pre-post gains for individual and class-wide comparisons, along with numbers of successful matches.
